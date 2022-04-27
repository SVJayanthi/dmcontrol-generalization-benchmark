import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac_lusr import SAC


def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std


class LUSR(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta

		self.lusr = m.LUSR(self.shared_cnn, self.head_cnn, obs_shape, args.num_shared_layers, args.num_filters).cuda()
		self.lusr_optimizer = torch.optim.Adam([
            {'params': list(self.lusr.shared_cnn.parameters()) + list(self.lusr.head_cnn.parameters())},
            {'params': list(self.lusr.linear_mu_projection.parameters()) + list(self.lusr.linear_classcode_projection.parameters())},
        ], lr=1e-2)

	def vae_loss(self, x, mu, logsigma, recon_x, beta=1):
		recon_loss = F.mse_loss(x, recon_x, reduction='mean')
		# kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
		# kl_loss = kl_loss / torch.numel(x)
		return recon_loss # + kl_loss * beta

	def forward_loss(self, x, beta):
		mu, classcode = self.lusr.encoder(x)
		# contentcode = reparameterize(mu, logsigma)
		shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

		latentcode1 = torch.cat([mu, shuffled_classcode], dim=1)
		latentcode2 = torch.cat([mu, classcode], dim=1)

		recon_x1 = self.lusr.decoder(latentcode1)*255
		recon_x2 = self.lusr.decoder(latentcode2)*255

		return self.vae_loss(x, mu, 0, recon_x1, beta) + self.vae_loss(x, mu, 0, recon_x2, beta)

	def backward_loss(self, x, device=None):
		mu, classcode = self.lusr.encoder(x)
		shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
		randcontent = torch.randn_like(mu) #.to(device)

		latentcode1 = torch.cat([randcontent, classcode], dim=1)
		latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

		recon_imgs1 = self.lusr.decoder(latentcode1).detach()*255
		recon_imgs2 = self.lusr.decoder(latentcode2).detach()*255

		cycle_mu1, cycle_classcode1 = self.lusr.encoder(recon_imgs1)
		cycle_mu2, cycle_classcode2 = self.lusr.encoder(recon_imgs2)

		# cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
		# cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

		bloss = F.l1_loss(cycle_mu1, cycle_mu2)
		return bloss

	def lusr_loss(self, x, beta):
		mu, classcode = self.lusr.encoder(x)
		# contentcode = reparameterize(mu, logsigma)
		shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

		latentcode1 = torch.cat([mu, shuffled_classcode], dim=1)
		latentcode2 = torch.cat([mu, classcode], dim=1)

		recon_x1 = self.lusr.decoder(latentcode1)
		recon_x2 = self.lusr.decoder(latentcode2)

		floss = self.vae_loss(x, mu, recon_x1, beta) + self.vae_loss(x, mu, recon_x2, beta)

		randcontent = torch.randn_like(mu).cuda()
		latentcode1 = torch.cat([randcontent, classcode], dim=1)
		latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

		recon_imgs1 = self.lusr.decoder(latentcode1).detach()
		recon_imgs2 = self.lusr.decoder(latentcode2).detach()

		cycle_mu1, cycle_classcode1 = self.lusr.encoder(recon_imgs1)
		cycle_mu2, cycle_classcode2 = self.lusr.encoder(recon_imgs2)

		bloss = F.l1_loss(cycle_mu1, cycle_mu2)
		return floss, bloss

	def update_lusr(self, obs, augments=9):
		imgs = obs.to('cuda', non_blocking=True)
		# for _ in range(augments):
		# 	imgs.append(augmentations.random_conv(obs.clone()))
		self.lusr_optimizer.zero_grad()
		floss = self.forward_loss(imgs, 10)
		floss = floss / imgs.shape[0]
		# print("FORWARD LOSS: ", floss.item())

		imgs = utils.cat(imgs, augmentations.random_conv(imgs.clone()))
		# backward circle
		# imgs = imgs.reshape(-1, *imgs.shape[2:])
		bloss = self.backward_loss(imgs)
		# print("BACKWARD LOSS: ", bloss.item())
		# floss, bloss = self.lusr_loss(imgs, 10)
		# floss = floss / imgs.shape[0]

		(floss + bloss).backward()
		# floss.backward()
		self.lusr_optimizer.step()
		# print("STEP")
		return floss, bloss

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_conv(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_init(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		floss, bloss = self.update_lusr(obs[:16])
		if L is not None:
			L.log('train/forward_loss', floss, step)
			L.log('train/backward_loss', bloss, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		return floss, bloss

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		# floss, bloss = self.update_lusr(obs[:16])
		# if L is not None:
		# 	L.log('train/forward_loss', floss, step)
		# 	L.log('train/backward_loss', bloss, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
