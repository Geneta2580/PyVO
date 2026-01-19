/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int R, int B, // Acc.
		const uint32_t* per_bucket_tile_offset,
		const uint32_t* bucket_to_tile,
		const float* sampled_T, const float* sampled_ar, // Acc.
		const float* sampled_ar_depth, const float* sampled_ar_accum, const float* sampled_ar_normal, const float* sampled_ar_reg, // Acc. 
		const float* sampled_M1, const float* sampled_M2, // Distortion.
		float focal_x, float focal_y,
		const float* bg_color,
		const float2* means2D,
		const float4* normal_opacity,
		const float* transMats,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const uint32_t* max_contrib,
		const float* pixel_colors, // Acc.
		const float* pixel_others, // Acc.
		const float* dL_dpixels,
		const float* dL_depths,
		const bool* pixel_mask,
		float * dL_dtransMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* scores);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
