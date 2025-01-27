import itertools
from functools import reduce, wraps
from operator import add

import numpy as np
import torch

from chai_lab.data.dataset.templates.fns import residue_constants as rc
from chai_lab.data.dataset.templates.fns.rigid_utils import Rotation, Rigid
from chai_lab.data.dataset.templates.fns.rigid_matrix_vector import Rigid3Array
from chai_lab.data.dataset.templates.fns.rotation_matrix import Rot3Array
from chai_lab.data.dataset.templates.fns.vector import Vec3Array
from chai_lab.data.dataset.templates.fns.tensor_utils import (
    tree_map,
    tensor_tree_map,
    batched_gather,
)

def atom37_to_frames(protein, is_multimer=False, eps=1e-8):
    aatype = torch.tensor(protein.aatype)
    all_atom_positions = torch.tensor(protein.atom_positions)
    all_atom_mask = torch.tensor(protein.atom_mask)

    if is_multimer:
        all_atom_positions = Vec3Array.from_array(all_atom_positions)

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(
        rc.chi_angles_mask
    )

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = (
        restype_rigidgroup_base_atom37_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
        )
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    if is_multimer:
        base_atom_pos = [batched_gather(
            pos,
            residx_rigidgroup_base_atom37_idx,
            dim=-1,
            no_batch_dims=len(all_atom_positions.shape[:-1]),
        ) for pos in all_atom_positions]
        base_atom_pos = Vec3Array.from_array(torch.stack(base_atom_pos, dim=-1))
    else:
        base_atom_pos = batched_gather(
            all_atom_positions,
            residx_rigidgroup_base_atom37_idx,
            dim=-2,
            no_batch_dims=len(all_atom_positions.shape[:-2]),
        )

    if is_multimer:
        point_on_neg_x_axis = base_atom_pos[:, :, 0]
        origin = base_atom_pos[:, :, 1]
        point_on_xy_plane = base_atom_pos[:, :, 2]
        gt_rotation = Rot3Array.from_two_vectors(
            origin - point_on_neg_x_axis, point_on_xy_plane - origin)

        gt_frames = Rigid3Array(gt_rotation, origin)
    else:
        gt_frames = Rigid.from_3_points(
            p_neg_x_axis=base_atom_pos[..., 0, :],
            origin=base_atom_pos[..., 1, :],
            p_xy_plane=base_atom_pos[..., 2, :],
            eps=eps,
        )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1

    if is_multimer:
        gt_frames = gt_frames.compose_rotation(
            Rot3Array.from_array(rots))
    else:
        rots = Rotation(rot_mats=rots)
        gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=aatype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    if is_multimer:
        ambiguity_rot = Rot3Array.from_array(residx_rigidgroup_ambiguity_rot)

        # Create the alternative ground truth frames.
        alt_gt_frames = gt_frames.compose_rotation(ambiguity_rot)
    else:
        residx_rigidgroup_ambiguity_rot = Rotation(
            rot_mats=residx_rigidgroup_ambiguity_rot
        )
        alt_gt_frames = gt_frames.compose(
            Rigid(residx_rigidgroup_ambiguity_rot, None)
        )

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    protein_dict = {}
    protein_dict["rigidgroups_gt_frames"] = gt_frames_tensor
    protein_dict["rigidgroups_gt_exists"] = gt_exists
    protein_dict["rigidgroups_group_exists"] = group_exists
    protein_dict["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    protein_dict["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return protein_dict

def get_backbone_frames(protein):
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.
    protein["backbone_rigid_tensor"] = protein["rigidgroups_gt_frames"][
        ..., 0, :, :
    ]
    protein["backbone_rigid_mask"] = protein["rigidgroups_gt_exists"][..., 0]

    return protein
