// SPDX-FileCopyrightText: Copyright (c) 2026 Patch
// SPDX-License-Identifier: Apache-2.0

// The SOMA character from soma.bin (export_character.py writes it from your MotionBricks checkout): a skinned
// mesh whose bones take recorded poses. Everything here lives in the runtime's world space (Z up, meters);
// viewer.js turns it Y up.
import * as THREE from "three";

const DTYPES = { float32: Float32Array, uint32: Uint32Array, uint16: Uint16Array };

export async function loadCharacterPack(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`character pack: HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "MBC1") throw new Error("character pack: bad magic");
  const headerLength = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const base = 8 + headerLength;
  const read = (entry) => new DTYPES[entry.dtype](buffer, base + entry.offset, entry.length);

  const geometries = header.parts.map((part) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(read(part.positions), 3));
    geometry.setIndex(new THREE.BufferAttribute(read(part.indices), 1));
    geometry.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(read(part.skin_index), 4));
    geometry.setAttribute("skinWeight", new THREE.BufferAttribute(read(part.skin_weight), 4));
    geometry.computeVertexNormals();
    return geometry;
  });

  const jointCount = header.joints.length;
  // Fingers draw in the file's relaxed pose, as upstream's renderer does: the model does not predict them.
  const override = header.override_poses.RelaxedHands || null;
  const mask = (header.override_masks && header.override_masks.Fingers) || new Array(jointCount).fill(0);
  const boneInverses = header.inverse_bind.map((t) =>
    new THREE.Matrix4().compose(
      new THREE.Vector3(t[0], t[1], t[2]),
      new THREE.Quaternion(t[3], t[4], t[5], t[6]),
      new THREE.Vector3(1, 1, 1),
    ),
  );
  return { header, geometries, jointCount, override, mask, boneInverses };
}

// One posable copy of the character. Each crowd character and each plan ghost is a rig.
export class Rig {
  constructor(pack, material, { castShadow = false } = {}) {
    this.pack = pack;
    this.group = new THREE.Group();
    this.bones = pack.header.joints.map((name) => {
      const bone = new THREE.Bone();
      bone.name = name;
      return bone;
    });
    pack.header.parents.forEach((parent, index) => {
      (parent >= 0 ? this.bones[parent] : this.group).add(this.bones[index]);
    });
    this.skeleton = new THREE.Skeleton(this.bones, pack.boneInverses.map((m) => m.clone()));
    this.meshes = pack.geometries.map((geometry) => {
      const mesh = new THREE.SkinnedMesh(geometry, material);
      mesh.bind(this.skeleton, new THREE.Matrix4());
      // The root bone carries the world placement, so the bind-pose bounds never fit.
      mesh.frustumCulled = false;
      mesh.castShadow = castShadow;
      this.group.add(mesh);
      return mesh;
    });
    this.applyPose(Float32Array.from(pack.header.reference.flat()));
  }

  // pose: Float32Array of joints x 7 (position, quaternion xyzw), local transforms, root first.
  applyPose(pose, offset = 0) {
    const { override, mask } = this.pack;
    for (let j = 0; j < this.bones.length; j++) {
      const bone = this.bones[j];
      if (override && mask[j] > 0.5) {
        const t = override[j];
        bone.position.set(t[0], t[1], t[2]);
        bone.quaternion.set(t[3], t[4], t[5], t[6]);
      } else {
        const i = offset + j * 7;
        bone.position.set(pose[i], pose[i + 1], pose[i + 2]);
        bone.quaternion.set(pose[i + 3], pose[i + 4], pose[i + 5], pose[i + 6]);
      }
    }
  }

  // Blend two streamed poses: positions lerp, rotations slerp.
  applyBlend(a, b, t) {
    const { override, mask } = this.pack;
    const qa = Rig._qa, qb = Rig._qb;
    for (let j = 0; j < this.bones.length; j++) {
      const bone = this.bones[j];
      if (override && mask[j] > 0.5) {
        const o = override[j];
        bone.position.set(o[0], o[1], o[2]);
        bone.quaternion.set(o[3], o[4], o[5], o[6]);
        continue;
      }
      const i = j * 7;
      bone.position.set(
        a[i] + (b[i] - a[i]) * t,
        a[i + 1] + (b[i + 1] - a[i + 1]) * t,
        a[i + 2] + (b[i + 2] - a[i + 2]) * t,
      );
      qa.set(a[i + 3], a[i + 4], a[i + 5], a[i + 6]);
      qb.set(b[i + 3], b[i + 4], b[i + 5], b[i + 6]);
      bone.quaternion.slerpQuaternions(qa, qb, t);
    }
  }

  set visible(value) {
    this.group.visible = value;
  }
}
Rig._qa = new THREE.Quaternion();
Rig._qb = new THREE.Quaternion();
