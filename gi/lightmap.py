"""Lightmap baking helpers.

Fixes for the six bake defects:

1. UV overlap detection (:func:`uv_overlap`) + chart repacking
   (:func:`repack_uvs`) so two faces can no longer share atlas texels.
2. :func:`texel_density` allocates atlas resolution from world-space size,
   so small props stop hogging the atlas.
3. :func:`dilate_lightmap` runs an arbitrary number of dilation passes,
   closing lightmap seams instead of bleeding only one texel.
4. :func:`cosine_hemisphere` replaces the fixed hemisphere directions with
   deterministic, stratified cosine-weighted sampling (low-discrepancy
   Hammersley points), which is what takes the noise out of indoor-corner AO.
5. :func:`cluster_lights` / :func:`cull_lights` count and cull lights per
   spatial cluster, so bake cost no longer grows linearly with every light
   in the scene.
6. :class:`IncrementalBaker` rebakes only clusters whose lights actually
   moved; unchanged clusters are served from cache.

Pure standard library, no third-party dependencies.
"""

import math


# ---------------------------------------------------------------------------
# Texel density
# ---------------------------------------------------------------------------

def texel_density(world_size, atlas_px):
    """Texels per world metre for a chart of ``world_size`` metres.

    The same physical resolution is then assigned to every object regardless
    of its size: a 4 m wall and a 2 m prop both get 128 texels/m when their
    atlas regions are 512 px and 256 px respectively.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive, got %r" % (world_size,))
    if atlas_px <= 0:
        raise ValueError("atlas_px must be positive, got %r" % (atlas_px,))
    return atlas_px / world_size


# ---------------------------------------------------------------------------
# UV overlap detection / repacking
# ---------------------------------------------------------------------------

def uv_overlap(rects):
    """Return the ``(i, j)`` index pairs whose UV rectangles overlap.

    Rectangles are axis-aligned ``(u0, v0, u1, v1)`` tuples. Intersection is
    tested on interiors only: charts that merely share an edge (touching at
    ``u1 == u0``) do not overlap, because shared edges do not alias texels.
    """
    pairs = []
    for i in range(len(rects)):
        u0a, v0a, u1a, v1a = rects[i]
        for j in range(i + 1, len(rects)):
            u0b, v0b, u1b, v1b = rects[j]
            if u0a < u1b and u0b < u1a and v0a < v1b and v0b < v1a:
                pairs.append((i, j))
    return pairs


def repack_uvs(rects, padding=1e-4):
    """Repack overlapping UV charts into a non-overlapping unit-square atlas.

    Charts keep their width/height (only their position changes), are placed
    with a shelf packer (tallest first), and the whole layout is uniformly
    scaled to fit ``[0, 1] x [0, 1]``. Returns new ``(u0, v0, u1, v1)``
    rectangles in the same order as the input, guaranteed to satisfy
    ``uv_overlap(result) == []``.
    """
    if not rects:
        return []

    charts = [
        (idx, max(0.0, r[2] - r[0]), max(0.0, r[3] - r[1]))
        for idx, r in enumerate(rects)
    ]
    # Tallest-first shelf packing gives a compact, deterministic layout.
    order = sorted(charts, key=lambda c: (-c[2], -c[1]))

    placed = {}
    x = y = 0.0
    shelf_height = 0.0
    max_x = 0.0
    for idx, w, h in order:
        if x > 0.0 and x + w > 1.0:
            # Wrap to a new shelf.
            y += shelf_height + padding
            x = 0.0
            shelf_height = 0.0
        placed[idx] = (x, y, x + w, y + h)
        x += w + padding
        shelf_height = max(shelf_height, h)
        max_x = max(max_x, x)
    total_height = y + shelf_height

    scale = min(1.0 / max_x if max_x > 0.0 else 1.0,
                1.0 / total_height if total_height > 0.0 else 1.0)

    result = [None] * len(rects)
    for idx, (u0, v0, u1, v1) in placed.items():
        result[idx] = (u0 * scale, v0 * scale, u1 * scale, v1 * scale)
    return result


# ---------------------------------------------------------------------------
# Seam dilation
# ---------------------------------------------------------------------------

def dilate_lightmap(data, width, height, passes):
    """Dilate valid texels into empty neighbours for ``passes`` rings.

    Each pass grows valid (non-zero) texels into their 8-connected empty
    (zero) neighbours using a standard 3x3 dilation kernel; existing valid
    texels are never overwritten, so baked colour is preserved. Multiple
    passes widen the dilation band, which keeps mip levels and bilinear
    filtering at UV seams from sampling empty texels (seam light leaking).
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if len(data) != width * height:
        raise ValueError("data length does not match width * height")

    image = list(data)
    for _ in range(max(0, passes)):
        previous = image
        expanded = list(previous)
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if previous[idx] != 0.0:
                    continue  # keep every valid texel untouched
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        if nx < 0 or nx >= width:
                            continue
                        value = previous[ny * width + nx]
                        if value != 0.0:
                            expanded[idx] = value
                            break
                    else:
                        continue
                    break
        image = expanded
    return image


# ---------------------------------------------------------------------------
# Cosine-weighted hemisphere sampling for AO
# ---------------------------------------------------------------------------

def _radical_inverse_vdc(integer):
    """Van der Corput radical inverse in base 2 -- deterministic low
    discrepancy sequence used to stratify the AO samples."""
    result = 0.0
    fraction = 0.5
    while integer > 0:
        result += (integer & 1) * fraction
        integer >>= 1
        fraction *= 0.5
    return result


def _normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0.0:
        raise ValueError("normal must not be the zero vector")
    return (v[0] / length, v[1] / length, v[2] / length)


def cosine_hemisphere(normal, sample_index, num_samples):
    """Deterministic cosine-weighted hemisphere sample around ``normal``.

    Sample ``sample_index`` of ``num_samples`` is mapped through a
    Hammersley low-discrepancy sequence and the standard cosine-weighted
    transform (PDF ``cos(theta) / pi``): samples cluster around the surface
    normal where the cosine contribution is strongest, which converges with
    far less noise than fixed hemisphere directions -- the fix for noisy AO
    in indoor corners. Output is a unit vector; sampling is fully
    deterministic, so identical indices always reproduce identical rays.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    index = sample_index % num_samples
    u1 = (index + 0.5) / num_samples
    u2 = _radical_inverse_vdc(index)

    # Cosine-weighted mapping: r = sqrt(u1), z = sqrt(1 - u1).
    radius = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    local_x = radius * math.cos(phi)
    local_y = radius * math.sin(phi)
    local_z = math.sqrt(max(0.0, 1.0 - u1))

    n = _normalize(normal)
    # Build an orthonormal basis (t, b, n) from an axis least parallel to n.
    if abs(n[0]) > abs(n[1]):
        inv = 1.0 / math.sqrt(n[0] * n[0] + n[2] * n[2])
        tangent = (-n[2] * inv, 0.0, n[0] * inv)
    else:
        inv = 1.0 / math.sqrt(n[1] * n[1] + n[2] * n[2])
        tangent = (0.0, n[2] * inv, -n[1] * inv)
    bitangent = (
        n[1] * tangent[2] - n[2] * tangent[1],
        n[2] * tangent[0] - n[0] * tangent[2],
        n[0] * tangent[1] - n[1] * tangent[0],
    )

    return (
        tangent[0] * local_x + bitangent[0] * local_y + n[0] * local_z,
        tangent[1] * local_x + bitangent[1] * local_y + n[1] * local_z,
        tangent[2] * local_x + bitangent[2] * local_y + n[2] * local_z,
    )


# ---------------------------------------------------------------------------
# Light clustering / culling
# ---------------------------------------------------------------------------

def cluster_lights(lights, clusters):
    """Count how many lights fall inside each spatial cluster.

    Lights are ``(x, y, z)`` points; clusters are axis-aligned bounding boxes
    ``(x0, y0, z0, x1, y1, z1)`` with inclusive bounds. Returns one count per
    cluster, letting the baker skip clusters with zero contributing lights
    instead of iterating every light for every surface.
    """
    counts = [0] * len(clusters)
    for light in lights:
        lx, ly, lz = light
        for c, (x0, y0, z0, x1, y1, z1) in enumerate(clusters):
            if x0 <= lx <= x1 and y0 <= ly <= y1 and z0 <= lz <= z1:
                counts[c] += 1
    return counts


def cull_lights(lights, clusters):
    """Return per-light visibility flags against the cluster set.

    A light is visible when it lies inside at least one cluster; lights
    culled here contribute to no surface and can be dropped from the bake
    entirely.
    """
    visible = [False] * len(lights)
    for i, (lx, ly, lz) in enumerate(lights):
        for (x0, y0, z0, x1, y1, z1) in clusters:
            if x0 <= lx <= x1 and y0 <= ly <= y1 and z0 <= lz <= z1:
                visible[i] = True
                break
    return visible


# ---------------------------------------------------------------------------
# Incremental rebake
# ---------------------------------------------------------------------------

class IncrementalBaker:
    """Per-cluster bake cache enabling incremental lightmap updates.

    Each cluster's baked block is cached together with a signature of the
    lights that affected it. On rebake, clusters whose signature is
    unchanged are served from cache (no work); moving a light only
    invalidates the clusters it belongs to, so the whole atlas is never
    rebaked.
    """

    def __init__(self):
        self._signatures = {}
        self._blocks = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _light_signature(lights):
        # Round so float jitter below practical precision does not invalidate
        # the cache, while real movement changes the signature.
        return tuple(
            tuple(round(component, 6) for component in light)
            for light in lights
        )

    def bake(self, cluster_id, lights, bake_fn):
        """Bake one cluster, reusing the previous block when lights match."""
        signature = self._light_signature(lights)
        if self._signatures.get(cluster_id) == signature:
            self.cache_hits += 1
            return self._blocks[cluster_id]

        block = bake_fn(lights)
        self._signatures[cluster_id] = signature
        self._blocks[cluster_id] = block
        self.cache_misses += 1
        return block

    def invalidate(self, cluster_id=None):
        """Drop cache for one cluster, or the whole atlas when omitted."""
        if cluster_id is None:
            self._signatures.clear()
            self._blocks.clear()
        else:
            self._signatures.pop(cluster_id, None)
            self._blocks.pop(cluster_id, None)
