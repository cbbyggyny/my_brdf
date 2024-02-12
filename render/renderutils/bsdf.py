import math
import torch

NORMAL_THRESHOLD = 0.1

################################################################################
# Vector utility functions
################################################################################

def _dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)

def _reflect(x, n):
    return 2*_dot(x, n)*n - x   #怎么感觉这样算出来的反射向量差了一个 - 号

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim= -1)

def _bend_normal(view_vec, smooth_nrm, geom_nrm, two_sided_shading):
    # Swap normal direction for backfacing surfaces
    if two_sided_shading:
        smooth_nrm = torch.where(_dot(geom_nrm, view_vec) > 0, smooth_nrm, -smooth_nrm)
        geom_nrm   = torch.where(_dot(geom_nrm, view_vec) > 0, geom_nrm, -geom_nrm)

    t = torch.clamp(_dot(view_vec, smooth_nrm) / NORMAL_THRESHOLD, min=0, max=1)
    return torch.lerp(geom_nrm, smooth_nrm, t)  # torch.lerp 函数，以线性插值的方式组合两个输入向量,t控制比例

def _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl):
    smooth_bitang = _safe_normalize(torch.cross(smooth_tng, smooth_nrm))    # 计算 N B T 空间
    if opengl:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] - smooth_bitang * perturbed_nrm[..., 1:2] + smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)   #？？？？？这一步计算没看懂
    else:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] + smooth_bitang * perturbed_nrm[..., 1:2] + smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)   #就差了第一个 + 号 ？
    return _safe_normalize(shading_nrm)

def bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
    smooth_nrm  = _safe_normalize(smooth_nrm)
    smooth_tng  = _safe_normalize(smooth_tng)
    view_vec    = _safe_normalize(view_pos - pos)
    shading_nrm = _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl)

################################################################################
# Simple lambertian diffuse BSDF
################################################################################

def bsdf_lambert(nrm, wi):
    return torch.clamp(_dot(nrm, wi), min=0.0) / math.pi

################################################################################
# Frostbite diffuse
################################################################################

def bsdf_frostbite(nrm, wi, wo, linearRoughness):
    wiDotN = _dot(wi, nrm)
    woDotN = _dot(wo, nrm)

    h = _safe_normalize(wo + wi)
    wiDotH = _dot(wi, h)

    energyBias = 0.5 * linearRoughness
    energyFactor = 1.0 - (0.51 / 1.51) * linearRoughness
    f90 = energyBias + 2.0 * wiDotH * wiDotH * linearRoughness
    f0 = 1.0

    wiScatter = bsdf_fresnel_shlick(f0, f90, wiDotN)
    woScatter = bsdf_fresnel_shlick(f0, f90, wiDotN)
    res = wiScatter * woScatter * energyFactor        #？？？？？？？？理论部分对应哪一块？
    return torch.where((wiDotN > 0.0)  & (woDotN > 0.0), res, torch.zeros_like(res))

################################################################################
# Phong specular, loosely based on mitsuba implementation 基于 Mitsuba 实现
################################################################################

def bsdf_phong(nrm, wo, wi, N):
    dp_r = torch.clamp(_dot(_reflect(wo, nrm), wi), min=0.0, max=1.0)
    dp_l = torch.clamp(_dot(nrm, wi), min=0.0, max=1.0)
    return (dp_r ** N) * dp_l *(N + 2) / (2 * math.pi)   #？？？？？？？？理论部分对应哪一块？

################################################################################
# PBR's implementation of GGX specular
################################################################################

specular_epsilon = 1e-4

def bsdf_fresnel_shlick(f0, f90, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0-specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0

def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0-specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * math.pi)

def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0-specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res

def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)

def bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    _alpha = torch.clamp(alpha, min=min_roughness * min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = _safe_normalize(wo + wi)
    woDotN = _dot(wo, nrm)
    wiDotN = _dot(wi, nrm)
    woDotH = _dot(wo, h)
    nDotH  = _dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))

def bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF):
    wo = _safe_normalize(view_pos - pos)
    wi = _safe_normalize(light_pos - pos)   #感觉定义的有点奇怪？？都是从物体指向视点/光源

    spec_str  = arm[..., 0:1]
    roughness = arm[..., 1:2]
    metallic  = arm[..., 2:3]
    ks = (0.04 * (1.0 - metallic) + kd * metallic) * (1 - spec_str)
    kd = kd * (1 - metallic)

    if BSDF == 0:
        diffuse = kd * bsdf_lambert(nrm, wi)
    else:
        diffuse = kd * bsdf_frostbite(nrm, wi, wo, roughness)
    specular = bsdf_pbr_specular(ks, nrm, wo, wi, roughness*roughness, min_roughness=min_roughness)
    return diffuse + specular