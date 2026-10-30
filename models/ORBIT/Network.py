import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet18_encoder import *
from models.resnet20_cifar import *
import torch.fft


class MYNET(nn.Module):

    def __init__(self, args, mode=None, trans=1):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder_q = resnet20(num_classes=self.args.moco_dim)
            self.encoder_k = resnet20(num_classes=self.args.moco_dim)
            self.num_features = 64


        if self.args.dataset in ['mini_imagenet', 'stl10', 'food101']:
            self.encoder_q = resnet18(False, args, num_classes=self.args.moco_dim)  # pretrained=False
            self.encoder_k = resnet18(False, args, num_classes=self.args.moco_dim)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200', 'gtsrb', 'dtd','fgvcaircraft']:
            self.encoder_q = resnet18(True, args, num_classes=self.args.moco_dim) 
            self.encoder_k = resnet18(True, args, num_classes=self.args.moco_dim)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes*trans, bias=False)
            
        self.K = self.args.moco_k
        self.m = self.args.moco_m
        self.T = self.args.moco_t
        
        if self.args.mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.register_buffer("queue", torch.randn(self.args.moco_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)
#for experiment T=I, uncomment this.
#    def replace_augmentation(self, feature, seed=None, use_identity_basis=True):
#        device = feature.device
#        g_T, g_perm, g_mix = (
#            [torch.Generator(device=device).manual_seed(seed + i) for i in range(3)]
#            if seed is not None else [None, None, None]
#        )
#    
#        restore = None
#        if feature.dim() == 4:
#            B, C, H, W = feature.shape
#            feature = F.adaptive_avg_pool2d(feature, 1).view(B, C)
#            restore = (B, C, H, W)
#        B, D = feature.shape
#    
#        if use_identity_basis:
#            T = torch.eye(D, device=device, dtype=feature.dtype)
#        else:
#            A = torch.randn(D, D, device=device, dtype=feature.dtype, generator=g_T)
#            Q, R = torch.linalg.qr(A)
#            d = torch.diagonal(R, 0)
#            T = Q * torch.where(d == 0, torch.ones_like(d), d.sign()).unsqueeze(0)
#    
#        f_t = feature @ T
#        perm = torch.randperm(B, generator=g_perm, device=device)
#        rot_lam = torch.rand(D, device=device, generator=g_mix)
#        f_rec = (rot_lam * f_t + (1 - rot_lam) * f_t[perm]) @ T.T
#    
#        if restore is not None:
#            Bc, C, H, W = restore
#            f_rec = f_rec.view(Bc, C, 1, 1).expand(Bc, C, H, W)
#        return f_rec
    def _vectorized_pair_mix(self, f_orig, idx, lam, generator=None):
            """Vectorized replacement for the old per-index Python loop.
    
            Pairs each index in `idx` with a random, *different* partner also in
            `idx`, and blends:
                out[i] = lam * f_orig[i] + (1 - lam) * f_orig[partner(i)]
    
            Why this matters for perf: the original loop called `.item()` once
            per element of `idx` to pick a random partner. Every `.item()` call
            forces a GPU->CPU synchronization -- for a batch of 64 with 32 odd
            indices, that's up to 32 forced stalls per `replace_augmentation`
            call. This version issues a small, FIXED number of vectorized ops
            (independent of batch size) and does zero `.item()` calls in the
            common case.
    
            Same guarantee as the original: no index is ever paired with itself.
            A random permutation of `idx` can produce fixed points (self-pairs);
            a few cheap vectorized cyclic-shift passes resolve almost all of
            them, and any rare residual (typically zero elements) is fixed
            exactly via a tiny fallback loop scoped to just the leftovers, not
            the whole batch.
            """
            device = f_orig.device
            n = idx.numel()
            out = f_orig.clone()
            if n < 2:
                return out
    
            local_perm = torch.randperm(n, generator=generator, device=device)
            partner = idx[local_perm]
    
            # Vectorized attempt to remove self-pairs (fixed points of the
            # permutation). A handful of cyclic-shift passes resolves the vast
            # majority of cases with no host-device sync at all.
            for _ in range(3):
                same = partner == idx
                if not same.any():
                    break
                local_perm = torch.where(same, torch.roll(local_perm, 1), local_perm)
                partner = idx[local_perm]
    
            same = partner == idx
            if same.any():
                # Rare exact fallback (typically 0 elements, occasionally 1-2 for
                # small batches). Guarantees no self-pairs, same guarantee the
                # original per-index loop gave -- just scoped to only the
                # leftover indices instead of the whole batch.
                residual = idx[same]
                for i in residual.tolist():
                    candidates = idx[idx != i]
                    j = candidates[torch.randint(0, len(candidates), (1,), generator=generator, device=device)]
                    partner[idx == i] = j
    
            out[idx] = lam * f_orig[idx] + (1 - lam) * f_orig[partner]
            return out

    def _make_random_orthogonal_basis(self, D, device, generator=None, dtype=torch.float32, identity=False):
        """
        Fast, Haar-uniform random D x D orthogonal matrix.

        Set identity=True to run the T=I ablation (isolates whether the
        orthonormal transform itself contributes anything beyond the
        permutation-blend + noise steps in `replace_augmentation`).

        Algorithm (Mezzadri 2006 -- the standard method for sampling from O(D)):
          1. Draw A ~ Gaussian(D, D).
          2. QR-decompose A = Q @ R.
          3. Multiply Q's columns by sign(diag(R)).

        Step 3 is not optional. Raw QR output is orthogonal but its distribution
        is skewed toward the identity-like orientation implied by the QR
        algorithm's sign convention; the sign correction removes that bias and
        makes Q uniform over O(D). Omitting it won't error or look wrong in
        any obvious way -- it'll just silently make T less random than intended,
        which is exactly the kind of bug that's invisible until you diff ablations.

        Performance: this is one call to torch.linalg.qr (cuSOLVER/LAPACK
        geqrf+orgqr under the hood) plus a couple of O(D^2) elementwise ops --
        no Python loops, no per-element .item() sync. It's O(D^3) FLOPs, which
        is unavoidable for generating a full D x D orthogonal matrix, but it's
        a single fused vectorized op rather than D sequential Householder steps,
        so it parallelizes well on GPU. Combined with the caching in
        `replace_augmentation` (recomputed only on shape/device/dtype change),
        the amortized cost is negligible.
        """
        if identity:
            return torch.eye(D, device=device, dtype=dtype)

        A = torch.randn(D, D, device=device, dtype=dtype, generator=generator)
        Q, R = torch.linalg.qr(A)
        d = torch.diagonal(R, 0)
        sign = d.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # d==0 has prob 0, but guard anyway
        Q = Q * sign.unsqueeze(0)
        return Q



    def replace_augmentation(self, feature, lamda=0.5, mix_lam=0.5, noise_std=0.02, seed=None):
        """
        Performance-optimized version.

        Unchanged from the previous rewrite (deliberately -- this is the part
        that actually makes the orthogonal basis T matter):
          - Per-dimension mixing weights (`rot_mix_lam`) in the rotated space.
            With a SCALAR mix weight here, `T @ (...) @ T.T` cancels exactly
            for any orthogonal T (including T = I) -- see derivation notes
            below. Do not "simplify" this back to a scalar for speed; that
            silently makes T's contribution vanish regardless of how it was
            constructed.
          - Decoupled RNG streams (`g_T`, `g_pair`, `g_perm`, `g_mix`,
            `g_noise`) keyed off `seed`, so a T=I vs. real-T ablation under a
            fixed seed is comparing like-for-like.

        Changed (performance only, no behavioral change to the math):
          1. Pairwise odd-index mixing is now vectorized (`_vectorized_pair_mix`)
             instead of a Python loop with a `.item()` call per element. Each
             `.item()` forces a GPU sync; this removes that entirely in the
             common case.
          2. Removed a redundant clone in both branches (previously cloned
             the feature tensor twice before the pairwise-mixing step).
          3. `_orth_basis` cache now also checks device and dtype, not just
             shape -- prevents a stale cross-device/cross-dtype basis from
             being silently reused (previously could error or misbehave if
             the model moved devices between calls).
          4. `_make_random_orthogonal_basis` is a proper Haar-uniform QR
             sampler (Mezzadri 2006) with sign-corrected Q, fully vectorized
             (single `torch.linalg.qr` call, no Python loop over dimensions),
             and takes `generator=` directly -- so the `g_T` seeding actually
             takes effect instead of silently falling back to global RNG.
          5. Both helpers are now proper bound methods (`self._vectorized_pair_mix`,
             `self._make_random_orthogonal_basis`) -- the previous NameError came
             from a stray duplicate def of `_vectorized_pair_mix` missing `self`
             in its signature, plus call sites in this method not using `self.`.

        Original derivation notes (unchanged from prior version):
          With a scalar mix_lam, the rotation-mixing step is exactly
          equivalent to plain mixup in the ORIGINAL feature space for any
          orthogonal T, because:
              f_rec = (lam*f_t + (1-lam)*f_t_p) @ T.T
                    = lam*(f @ T @ T.T) + (1-lam)*(f_p @ T @ T.T)
                    = lam*f + (1-lam)*f_p        [since T @ T.T = I]
          i.e. T drops out completely. The per-dimension weight vector
          `rot_mix_lam` breaks this cancellation because the mix is no longer
          a uniform linear combination across all rotated coordinates -- it's
          genuinely different in the rotated basis than in the original one.
        """
        device = feature.device

        if seed is not None:
            g_T     = torch.Generator(device=device).manual_seed(seed)
            g_pair  = torch.Generator(device=device).manual_seed(seed + 1)
            g_perm  = torch.Generator(device=device).manual_seed(seed + 2)
            g_mix   = torch.Generator(device=device).manual_seed(seed + 3)
            g_noise = torch.Generator(device=device).manual_seed(seed + 4)
        else:
            g_T = g_pair = g_perm = g_mix = g_noise = None

        # ---------------- fine-grained branch ----------------
        if False:#hasattr(self, "is_fine_grained") and self.is_fine_grained:
            odd_idx = torch.arange(1, feature.size(0), 2, device=device)
            return self._vectorized_pair_mix(feature, odd_idx, lamda, generator=g_pair)
        else:
        # ---------------- main branch ----------------
          restore = None
          if feature.dim() == 4:
              B, C, H, W = feature.shape
              feature = F.adaptive_avg_pool2d(feature, 1).view(B, C)
              restore = (B, C, H, W)
  
          B, D = feature.shape
  
          # --- pre-rotation odd-index pairwise mixing (vectorized, see above) ---
          odd_idx = torch.arange(1, B, 2, device=device)
          f = self._vectorized_pair_mix(feature, odd_idx, mix_lam, generator=g_pair)
  
          # --- orthogonal basis (cached per feature dim / device / dtype) ---
          need_new_basis = (
              not hasattr(self, "_orth_basis")
              or self._orth_basis.shape[0] != D
              or self._orth_basis.device != device
              or self._orth_basis.dtype != feature.dtype
          )
          if need_new_basis:
              self._orth_basis = self._make_random_orthogonal_basis(
                  D, device, generator=g_T, dtype=feature.dtype
              )
          T = self._orth_basis
  
          # --- rotate ---
          f_t = f @ T
  
          # --- sample-pairing permutation (rows), from its own RNG stream ---
          perm = torch.randperm(B, generator=g_perm, device=device) \
              if g_perm is not None else torch.randperm(B, device=device)
          f_t_p = f_t[perm]
  
          # --- PER-DIMENSION mixing weights: this is what makes T non-cancelling ---
          # See derivation notes in the docstring -- keep this a vector, not a scalar.
          rot_mix_lam = torch.rand(D, device=device, generator=g_mix) if g_mix is not None \
              else torch.rand(D, device=device)
  
          f_mix = rot_mix_lam * f_t + (1 - rot_mix_lam) * f_t_p     # elementwise per rotated dim
          f_rec = f_mix @ T.T
  
          # --- blend, add noise, renormalize ---
          f_out = 0.5 * f + 0.5 * f_rec
          if g_noise is not None:
              noise = torch.randn(f_out.shape, generator=g_noise, device=device) * noise_std
          else:
              noise = torch.randn_like(f_out) * noise_std
          f_out = f_out + noise
          #f_out = f_out / (f_out.norm(dim=1, keepdim=True) + 1e-9)
  
          if restore is not None:
              Bc, C, H, W = restore
              f_out = f_out.view(Bc, C, 1, 1).expand(Bc, C, H, W)
  
          return f_out
#    def replace_augmentation(self, feature, lamda=0.5, mix_lam=0.5, noise_std=0.02):
#        # works in same spirit, but no DFT / FFT kernel
#    
#        if hasattr(self, "is_fine_grained") and self.is_fine_grained:
#            feature = feature.clone()
#            o = torch.arange(1, feature.size(0), 2)
#    
#            for i in o:
#                j = o[o != i][torch.randint(len(o)-1, (1,))]
#                feature[i] = lamda * feature[i] + (1 - lamda) * feature[j]
#            return feature
#    
#        restore = None
#        mix_lam = 0.5
#    
#        if feature.dim() == 4:
#            B, C, H, W = feature.shape
#            feature = F.adaptive_avg_pool2d(feature, 1).view(B, C)
#            restore = (B, C, H, W)
#    
#        B, D = feature.shape
#        f = feature.clone()
#        odd_idx = torch.arange(1, B, 2)
#    
#        for i in odd_idx:
#            other = odd_idx[odd_idx != i]
#            if len(other) == 0:
#                continue
#            j = other[torch.randint(0, len(other), (1,)).item()]
#            f[i] = mix_lam * f[i] + (1 - mix_lam) * f[j]
#    
#        # --- non-FFT decorrelating basis ---
#        #use_identity = getattr(self, "use_identity_basis", False)
##        if not hasattr(self, "_orth_basis") or self._orth_basis.shape[0] != D \
##                or getattr(self, "_orth_basis_is_identity", None) != use_identity:
##            self._orth_basis = self._make_random_orthogonal_basis(D, feature.device, identity=use_identity)
#            #self._orth_basis_is_identity = use_identity
#        if not hasattr(self, "_orth_basis") or self._orth_basis.shape[0] != D:
#            self._orth_basis = self._make_random_orthogonal_basis(D, feature.device)
#    
#        T = self._orth_basis
#        f_t = f @ T
#        f_t_p = f_t[torch.randperm(B)]   # shuffle columns (features), not rows (samples)$f_t[torch.randperm(B)]
#        f_mix = mix_lam * f_t + (1 - mix_lam) * f_t_p
#        f_rec = f_mix @ T.T
#    
#        f_out = 0.5 * f + 0.5 * f_rec
#        f_out += torch.randn_like(f_out) * noise_std
#        f_out = f_out / (f_out.norm(dim=1, keepdim=True) + 1e-9)
#    
#        if restore is not None:
#            B, C, H, W = restore
#            f_out = f_out.view(B, C, 1, 1).expand(B, C, H, W)
#    
#        return f_out
##    def replace_augmentation(self, feature, lamda=0.5):
##        feature = feature.clone()  
##        odd_indices = torch.arange(1, feature.shape[0], 2)
##        for idx in odd_indices:
##            other_odd_indices = odd_indices[odd_indices != idx]
##            random_odd_idx = other_odd_indices[torch.randint(0, len(other_odd_indices), (1,)).item()]
##            original_embedding = feature[idx].clone()
##            random_embedding = feature[random_odd_idx].clone()
##            new_embedding = lamda * original_embedding + (1 - lamda) * random_embedding
##            feature[idx] = new_embedding
##        return feature
#
##    def replace_augmentation(self, feature, lamda=0.5):
##        feature = feature.clone()
##        perm = torch.randperm(feature.shape[0], device=feature.device)
##        feature = lamda * feature + (1 - lamda) * feature[perm]
##        return feature
#    def replace_augmentation(self, feature, lamda=0.5, mix_lam=0.5, noise_std=0.02):
##        if hasattr(self, "is_fine_grained") and self.is_fine_grained:
##          feature = feature.clone()  
##          odd_indices = torch.arange(1, feature.shape[0], 2)
##          for idx in odd_indices:
##              other_odd_indices = odd_indices[odd_indices != idx]
##              random_odd_idx = other_odd_indices[torch.randint(0, len(other_odd_indices), (1,)).item()]
##              original_embedding = feature[idx].clone()
##              random_embedding = feature[random_odd_idx].clone()
##              new_embedding = lamda * original_embedding + (1 - lamda) * random_embedding
##              feature[idx] = new_embedding
##          return feature
#          
#      if getattr(self, "is_fine_grained", False):
##        f = feature.clone()
##        odd = torch.arange(1, f.size(0), 2, device=f.device)
##        
##        f[odd] = lamda * f[odd] + (1 - lamda) * f[odd[(torch.arange(len(odd), device=f.device) + torch.randint(1, len(odd), (len(odd),), device=f.device)) % len(odd)]]
##        return f
#          feature = feature.clone()
#          o = torch.arange(1, feature.size(0), 2)
#          
#          for i in o:
#              j = o[o != i][torch.randint(len(o)-1, (1,))]
#              feature[i] = lamda * feature[i] + (1 - lamda) * feature[j]
#          return feature
#      else:
#          mix_lam=0.5
#          restore = None  # <-- add this line
#          if feature.dim() == 4:
#              B, C, H, W = feature.shape
#              feature = F.adaptive_avg_pool2d(feature, 1).view(B, C)
#              restore = (B, C, H, W)
#      
#          B, D = feature.shape
#          feature = feature.clone()
#          odd_indices = torch.arange(1, B, 2)
#          for idx in odd_indices:
#              other = odd_indices[odd_indices != idx]
#              if len(other) == 0:
#                  continue
#              rand_idx = other[torch.randint(0, len(other), (1,)).item()]
#              feature[idx] = mix_lam * feature[idx] + (1 - mix_lam) * feature[rand_idx]
#          ffted = torch.fft.fft(feature, dim=1)
#          ffted_perm = ffted[torch.randperm(B)]
#      
#          mag = torch.abs(ffted)
#          phase = torch.angle(ffted)
#          mag_p = torch.abs(ffted_perm)
#          phase_p = torch.angle(ffted_perm)
#      
#          mag_mix =  mag_p
#          phase_mix = phase_p
#      
#          fft_mix = mag_mix * torch.exp(1j * phase_mix)
#          feature_freq = torch.fft.ifft(fft_mix, dim=1).real
#      
#          feature = 0.5 * feature + 0.5 * feature_freq
#          feature += torch.randn_like(feature) * noise_std
#          feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-9)
#          if restore is not None:
#              B, C, H, W = restore
#              feature = feature.view(B, C, 1, 1).expand(B, C, H, W)
#      
#          return feature
#
#
##  


    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess):
    
        if base_sess:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        else:
            for k, v in self.encoder_q.named_parameters():
                if k.startswith('fc') or k.startswith('layer4') or k.startswith('layer3'):
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[k].data * self.m + v.data * (1. - self.m)
                    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
 
        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains:]
            self.label_queue[ptr:] = labels[ :batch_size - remains]
            self.label_queue[ :remains] = labels[batch_size - remains:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T # this queue is feature queue
            self.label_queue[ptr:ptr + batch_size] = labels        
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
    
    def forward_metric(self, x):
        x, _ = self.encode_q(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x # joint, contrastive

    def encode_q(self, x):
        x, y = self.encoder_q(x)
        x = self.replace_augmentation(x)#, lamda=0.5)
        y = self.replace_augmentation(y)#, lamda=0.5)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y
    
    def encode_k(self, x):
        x, y = self.encoder_k(x)
        x = self.replace_augmentation(x)#, lamda=0.5)
        y = self.replace_augmentation(y)#, lamda=0.5)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def forward(self, im_cla, im_q=None, im_k=None, labels=None, im_q_small=None, base_sess=True, 
                last_epochs_new=False):
        if self.mode != 'encoder':
            if im_q == None:
                x = self.forward_metric(im_cla)
                return x
            else:
                b = im_q.shape[0]
                clasification_logits = self.forward_metric(im_cla)
                _, q = self.encode_q(im_q)

                q = nn.functional.normalize(q, dim=1)
                feat_dim = q.shape[-1]
                q = q.unsqueeze(1) # bs x 1 x dim

                if im_q_small is not None:
                    _, q_small = self.encode_q(im_q_small)
                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_whole = q
                l_pos_global = (q_whole * k.unsqueeze(1)).sum(2).view(-1, 1)
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_whole.view(-1, feat_dim), self.queue.clone().detach()])
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim), self.queue.clone().detach()])

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda()
                # find same label images from label queue
                # for the query with -1, all 
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets], dim=1)
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1], dim=0)

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)
                
                return clasification_logits, logits_global, logits_small, targets_global, targets_small          
        
        elif self.mode == 'encoder':
            x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')
    
    
    def update_fc(self,dataloader,class_list,transform,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b 
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            data, _ =self.encode_q(data)
            data.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list)*m, self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m)
            

    def update_fc_avg(self,data,labels,class_list, m):
        new_fc=[]
        for class_index in class_list:
            for i in range(m):
                index = class_index*m + i
                data_index=(labels==index).nonzero().squeeze(-1)
                embedding=data[data_index]
                proto=embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
