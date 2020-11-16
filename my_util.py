import torch

def ClippedPPOLoss(advantage, log_pi, log_old_pi, eps):
  ratio = torch.exp(log_pi - log_old_pi)
  clipped_ratio = torch.clip(ratio, 1-eps, 1+eps)
  clipped_reward = torch.min(ratio*advantage, clipped_ratio*advantage)
  return clipped_reward.mean() 

def ValueFunctionLoss(new_value, old_value):
  return ((new_value-old_value)**2).mean()

def ClippedValueFunctionLoss(value: torch.Tensor, sampled_value: torch.Tensor, sampled_return: torch.Tensor, clip: float):
  clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)
  vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
  return 0.5 * vf_loss.mean()