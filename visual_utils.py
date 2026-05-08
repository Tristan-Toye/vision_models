import numpy as np
import matplotlib as mpl
import pygame
from pettingzoo.utils import BaseWrapper
from pettingzoo.butterfly.knights_archers_zombies.src.zombie import Zombie
from pettingzoo.butterfly.knights_archers_zombies import knights_archers_zombies as KAZEnvModule
from gymnasium import spaces


zombie_mask_prob = 0.2


def photometric_jitter_transform(gamma_range=(0.8, 1.3), brightness=(0.9, 1.1), contrast=(0.9, 1.1), prob=1.0):
    def transform(frame_whc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if prob < 1.0 and rng.random() > prob:
            return frame_whc
        img = frame_whc.astype(np.float32)

        g = rng.uniform(*gamma_range)
        b = rng.uniform(*brightness)
        c = rng.uniform(*contrast)

        # gamma (normalize to 0..1, apply pow, back to 0..255)
        x = np.clip(img / 255.0, 0.0, 1.0) ** g
        x = x * 255.0

        # contrast around mid-gray + brightness
        x = (x - 128.0) * c + 128.0
        x = x * b

        frame_whc[:] = np.clip(x, 0, 255).astype(np.uint8)
        return frame_whc
    return transform

def heat_haze_warp_transform(amplitude_px=4.0, wavelength_px=90.0, prob=1.0):
    amplitude_px = float(amplitude_px)
    wavelength_px = float(wavelength_px)

    def transform(frame_whc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if prob < 1.0 and rng.random() > prob:
            return frame_whc

        W, H = frame_whc.shape[0], frame_whc.shape[1]

        # coordinate grids
        xs = np.arange(W, dtype=np.float32)[:, None]
        ys = np.arange(H, dtype=np.float32)[None, :]

        # random phases per call
        phx = rng.uniform(0, 2*np.pi)
        phy = rng.uniform(0, 2*np.pi)

        # displacement fields (smooth, low-frequency)
        dx = amplitude_px * np.sin((ys / wavelength_px) * 2*np.pi + phx)
        dy = amplitude_px * np.sin((xs / wavelength_px) * 2*np.pi + phy)

        # sample with nearest neighbor (fast, OK for small amp)
        src_x = np.clip(np.round(xs + dx).astype(np.int32), 0, W - 1)
        src_y = np.clip(np.round(ys + dy).astype(np.int32), 0, H - 1)

        warped = frame_whc[src_x, src_y, :]
        frame_whc[:] = warped
        return frame_whc

    return transform





def add_clouds_transform(
    intensity: float = 0.35,      # overall opacity multiplier
    coverage: float = 0.35,       # 0..1: how much of the screen becomes cloudy
    scale: int = 32,              # larger => bigger blobs, smaller => more detailed
    blur_passes: int = 2,         # softens edges
    tint=(235, 235, 235),         # cloud color
    prob: float = 1.0,            # apply sometimes
    seed_static: bool = False,    # if True, same clouds every call (uses first RNG draw)
):
    """
    Returns a transform(frame_whc, rng)->frame_whc that overlays soft clouds.

    Parameters:
      - intensity: 0..1 controls opacity of clouds
      - coverage: 0..1 controls how much of the screen is cloud (higher => more clouds)
      - scale: approximate blob size in pixels (higher => larger clouds)
      - blur_passes: number of box-blur passes on the noise field
      - tint: RGB tuple for cloud color
      - seed_static: if True, clouds pattern stays fixed over time (nice for static weather)
    """
    intensity = float(intensity)
    coverage = float(coverage)
    scale = int(max(4, scale))
    blur_passes = int(max(0, blur_passes))
    tint = np.array(tint, dtype=np.float32)
    prob = float(prob)

    cached_noise = {"field": None}

    def _box_blur_2d(a: np.ndarray) -> np.ndarray:
        # fast-ish separable 3x3 blur using rolls
        s = (
            a
            + np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1)
            + np.roll(np.roll(a, 1, 0), 1, 1)
            + np.roll(np.roll(a, 1, 0), -1, 1)
            + np.roll(np.roll(a, -1, 0), 1, 1)
            + np.roll(np.roll(a, -1, 0), -1, 1)
        )
        return s / 9.0

    def _upsample_nn(field: np.ndarray, W: int, H: int) -> np.ndarray:
        # field is (w0, h0) in WH layout; upsample to (W,H)
        w0, h0 = field.shape
        x_idx = (np.linspace(0, w0 - 1, W)).astype(np.int32)
        y_idx = (np.linspace(0, h0 - 1, H)).astype(np.int32)
        return field[np.ix_(x_idx, y_idx)]

    def transform(frame_whc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if prob < 1.0 and rng.random() > prob:
            return frame_whc

        img = frame_whc
        W, H = img.shape[0], img.shape[1]

        # Create (or reuse) a low-res noise field
        if seed_static and cached_noise["field"] is not None:
            low = cached_noise["field"]
        else:
            w0 = max(2, W // scale)
            h0 = max(2, H // scale)
            low = rng.random((w0, h0), dtype=np.float32)
            # make it smoother and more cloud-like
            for _ in range(max(1, blur_passes)):
                low = _box_blur_2d(low)
            if seed_static:
                cached_noise["field"] = low

        # Upsample to full resolution (still in WH layout)
        field = _upsample_nn(low, W, H)

        # Additional blur at full-res for soft edges
        for _ in range(blur_passes):
            field = _box_blur_2d(field)

        # Normalize and turn into a cloud mask
        # We want only the top "coverage" fraction to be cloudy
        # coverage=0.35 => keep top 35% brightest parts
        thresh = np.quantile(field, 1.0 - np.clip(coverage, 0.0, 1.0))
        mask = np.clip((field - thresh) / max(1e-6, (field.max() - thresh)), 0.0, 1.0)

        # Soften mask (more natural clouds)
        mask = mask * mask  # bias toward softer edges

        # Alpha blend cloud tint over img: img = img*(1-a) + tint*a
        a = np.clip(mask * intensity, 0.0, 1.0).astype(np.float32)  # (W,H)
        img_f = img.astype(np.float32)

        # broadcast a to (W,H,1)
        a3 = a[:, :, None]
        out = img_f * (1.0 - a3) + tint[None, None, :] * a3

        img[:, :, :] = np.clip(out, 0, 255).astype(np.uint8)
        return img

    return transform




def add_random_stars_transform(
    n_stars: int = 150,
    size_min: int = 1,
    size_max: int = 4,
    brightness_min: int = 200,
    brightness_max: int = 255,
    colored: bool = False,
    alpha: float = 1.0,
    prob: float = 1.0,
):
    """
    Returns a transform(frame_whc, rng) -> frame_whc that draws random stars.

    frame_whc: uint8 array shaped (W, H, 3).
    rng: np.random.Generator.

    Notes:
      - alpha blends stars over the current pixels: 1.0 = overwrite, 0.5 = mix.
      - colored=False makes white/yellowish stars; True makes random pastel-ish colors.
      - prob lets you apply stars only sometimes (e.g., flicker effects).
    """
    n_stars = int(n_stars)
    size_min = int(size_min)
    size_max = int(size_max)
    brightness_min = int(brightness_min)
    brightness_max = int(brightness_max)
    alpha = float(alpha)
    prob = float(prob)

    def _blend(dst_rgb_u8: np.ndarray, src_rgb_u8: np.ndarray, a: float) -> np.ndarray:
        if a >= 1.0:
            return src_rgb_u8
        if a <= 0.0:
            return dst_rgb_u8
        # Blend in float then clamp
        out = (dst_rgb_u8.astype(np.float32) * (1.0 - a) + src_rgb_u8.astype(np.float32) * a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_star(img: np.ndarray, x: int, y: int, r: int, color: np.ndarray):
        """
        img is (W,H,3). x indexes width axis, y indexes height axis.
        Draw a sparkle: center + cross arms + (optional) diagonals depending on r.
        """
        W, H = img.shape[0], img.shape[1]

        # helper to plot a point with blending
        def plot(px, py):
            if 0 <= px < W and 0 <= py < H:
                img[px, py, :] = _blend(img[px, py, :], color, alpha)

        # center
        plot(x, y)

        # arms
        for d in range(1, r + 1):
            plot(x + d, y)
            plot(x - d, y)
            plot(x, y + d)
            plot(x, y - d)

        # diagonals for bigger stars
        if r >= 2:
            for d in range(1, r):
                plot(x + d, y + d)
                plot(x - d, y - d)
                plot(x + d, y - d)
                plot(x - d, y + d)

    def transform(frame_whc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if prob < 1.0 and rng.random() > prob:
            return frame_whc

        img = frame_whc  # modify in place (wrapper already passes a copy typically)
        W, H = img.shape[0], img.shape[1]

        # guard
        if W <= 0 or H <= 0 or n_stars <= 0:
            return img

        for _ in range(n_stars):
            x = int(rng.integers(0, W))
            y = int(rng.integers(0, H))
            r = int(rng.integers(size_min, size_max + 1))

            b = int(rng.integers(brightness_min, brightness_max + 1))

            if colored:
                # pastel-ish random colors (biased toward bright)
                c = rng.integers(0, 256, size=(3,), dtype=np.int32)
                c = (c + b) // 2
                color = np.clip(c, 0, 255).astype(np.uint8)
            else:
                # white/yellowish
                # slight warm tint: [b, b, ~0.9b]
                color = np.array([b, b, int(b * 0.9)], dtype=np.uint8)

            _draw_star(img, x, y, r, color)

        return img

    return transform


def gaussian_noise_transform(std: float = 5.0, prob: float = 1.0, clip_low: int = 0, clip_high: int = 255):
    """
    Returns a transform(frame_whc)->frame_whc that adds Gaussian noise.
    frame_whc: np.uint8 array shaped (W, H, 3) like pygame.surfarray.pixels3d(surface).
    """
    std = float(std)
    prob = float(prob)
    clip_low = int(clip_low)
    clip_high = int(clip_high)

    def _transform(frame_whc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if prob < 1.0 and rng.random() > prob:
            return frame_whc

        x = frame_whc.astype(np.float32, copy=False)
        if std > 0:
            x = x + rng.normal(0.0, std, size=x.shape).astype(np.float32)
        x = np.clip(x, clip_low, clip_high)
        return x.astype(np.uint8, copy=False)

    return _transform


def compose(*tfs):
    def transform(frame_whc, rng):
        for tf in tfs:
            frame_whc = tf(frame_whc, rng)
        return frame_whc
    return transform





stars_tf = add_random_stars_transform(
    n_stars=200,
    size_min=1,
    size_max=4,
    colored=False,
    alpha=0.9,
)

noise_tf = gaussian_noise_transform(40,1)


clouds_tf = add_clouds_transform(
    intensity=1,
    coverage=0.30,
    scale=40,
    blur_passes=10,
    tint=(240, 240, 245),
    seed_static=False,   # clouds stay consistent over time
)


distortion_level = 0
minimal_tf = compose(
    photometric_jitter_transform(gamma_range=(0.85, 1.25), brightness=(0.95, 1.05), contrast=(0.9, 1.1)),
    heat_haze_warp_transform(amplitude_px=10.0, wavelength_px=110.0),
    clouds_tf,
    stars_tf
    )

def set_distortion_level(level=5):
    global minimal_tf
    global distortion_level
    distortion_level = level
    tfs = []
    if level > 0:
        tfs.append(stars_tf)
    if level > 1:
        tfs.append(clouds_tf)
    if level > 2: 
        pass  # change color of zombies (done in ZombieDisguised class)
    if level > 3:
        pass  # change pixels of zombies (done in ZombieDisguised class)
    if level > 4:
        tfs.append(heat_haze_warp_transform(amplitude_px=10.0, wavelength_px=110.0))
    if level > 5:
        tfs.append(photometric_jitter_transform(gamma_range=(0.85, 1.25), brightness=(0.95, 1.05), contrast=(0.9, 1.1)))
    minimal_tf = compose(*tfs)


class VisualWrapper(BaseWrapper):
    """
    Global postprocess hook for KAZ that affects:
      - the whole arena "screen"
      - human visuals
      - pixel observations (cropped) *because we write back into env.screen*

    You only customize: transform(frame_whc, rng) -> frame_whc.

    Notes:
      - frame_whc is (W, H, 3) uint8 (pygame pixels3d layout)
      - To make agent obs reflect transforms, use vector_state=False.
      - Underlying env.render_mode is forced to "rgb_array" to disable KAZ's internal human window.
      - Wrapper exposes render_mode="human" outward and renders once per full cycle.
    """

    def __init__(self, env, caption="KAZ (Transformed)"):
        super().__init__(env)
        self.transform = minimal_tf
        self.caption = caption

        # reproducible RNG if env provides it
        self._rng = getattr(env, "np_random", None) or np.random.default_rng()

        # wrapper owns the window
        self.render_mode = "human"
        self.env.render_mode = "rgb_array"

        self._pg_inited = False
        self._display = None
        self._clock = None

        # Cache for the transformed pixels (W,H,3)
        self._frame_whc = None
        self._cached_at_frames = None  # env.frames

        # Overwrite the generation of zombies
        # kaz_env = self.env.env.env
        # Monkeypatch with our custom zombie class
        KAZEnvModule.Zombie = ZombieDisguised
        # import pdb
        # pdb.set_trace()

    def reset(self, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)

        # Ensure env.screen exists (KAZ creates it in render() if missing)
        _ = self.env.render()

        # Build first transformed frame + show it
        self._refresh_transformed_frame(force=True)
        if self.render_mode == "human":
            self._show()
        return out

    def step(self, action):
        # Match KAZ cadence: render only once per full cycle
        selector = getattr(self.env, "_agent_selector", None)
        was_last = bool(selector.is_last()) if selector is not None else True

        out = self.env.step(action)

        self._refresh_transformed_frame(force=True)
        if self.render_mode == "human" and was_last:
            # After last agent, env has drawn the new frame into env.screen
            self._show()
        return out


    def observation_space(self, agent):
        # full screen returned by observe()
        w, h = self.env.screen.get_size()  # pygame screen size
        return spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

    def observe(self, agent):
        # IMPORTANT: KAZ pixel observations crop from env.screen.
        # Since we write transformed pixels back into env.screen, we can defer to env.observe().
        # return self.env.observe(agent)
        # Overwrite entire function to return full game view
        if self.vector_state:
            raise ValueError("You cannot use vector_state")
        screen = pygame.surfarray.pixels3d(self.screen)
        cropped = np.array(screen)
        return np.swapaxes(cropped, 1, 0)

    def render(self):
        # Optional explicit render: refresh and show/return
        self._refresh_transformed_frame(force=True)
        if self.render_mode == "human":
            self._show()
            return None
        return self._frame_hwc()

    def close(self):
        try:
            if self._pg_inited:
                pygame.display.quit()
                pygame.quit()
        except Exception:
            pass
        return super().close()

    # ---------- internals ----------

    def _refresh_transformed_frame(self, force: bool):
        cur_frames = getattr(self.env, "frames", None)
        if (not force) and (self._cached_at_frames == cur_frames) and (self._frame_whc is not None):
            return

        # Make sure screen exists and is current
        if getattr(self.env, "screen", None) is None:
            _ = self.env.render()

        # Grab clean pixels (W,H,3) and transform
        clean_whc = pygame.surfarray.pixels3d(self.env.screen).copy()
        transformed_whc = self.transform(clean_whc, self._rng)

        # Write back so KAZ observe() crops from transformed world
        pygame.surfarray.blit_array(self.env.screen, transformed_whc)

        self._frame_whc = transformed_whc
        self._cached_at_frames = cur_frames

    def _ensure_display(self, w: int, h: int):
        if not self._pg_inited:
            pygame.init()
            self._pg_inited = True
        if self._display is None:
            self._display = pygame.display.set_mode((w, h))
            pygame.display.set_caption(self.caption)
            self._clock = pygame.time.Clock()

    def _show(self):
        if self._frame_whc is None:
            return
        w, h = self._frame_whc.shape[0], self._frame_whc.shape[1]
        self._ensure_display(w, h)

        surf = pygame.surfarray.make_surface(self._frame_whc[:, :, :3])  # expects (W,H,3)
        self._display.blit(surf, (0, 0))
        pygame.display.flip()

        fps = int(getattr(self.env, "metadata", {}).get("render_fps", 0) or 0)
        self._clock.tick(fps)



class ZombieDisguised(Zombie):
    def __init__(self, randomizer):
        super().__init__(randomizer)
        if distortion_level > 2:
            self.disguise_zombie_color()
        if distortion_level > 3:
            self.disguise_zombie_pixels()

    def disguise_zombie_pixels(self):
        surface = self.image
        pixels = pygame.surfarray.pixels3d(surface)
        # Replace randomly selected pixels with a random color
        mask = np.random.rand(*pixels.shape[:2]) < zombie_mask_prob
        random_colors = np.random.randint(
            0, 256, pixels.shape, dtype=np.uint8
        )
        pixels[mask] = random_colors[mask]
        del pixels

    def disguise_zombie_color(self):
        surface = self.image
        pixels = pygame.surfarray.pixels3d(surface)
        # Replace green pixels
        target_hue = np.random.rand()
        img = np.transpose(pixels, (1, 0, 2))
        imgn = img / 255.
        hsv = mpl.colors.rgb_to_hsv(imgn)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        green_mask = (
            (h > 0.15) & (h < 0.60) &   # green hue range
            (s > 0.2)                   # avoid gray pixels
        )
        h[green_mask] = target_hue
        hsv[..., 0] = h
        new_img = (mpl.colors.hsv_to_rgb(hsv) * 255).astype(img.dtype)
        pixels[:] = np.transpose(new_img, (1, 0, 2))
        del pixels

