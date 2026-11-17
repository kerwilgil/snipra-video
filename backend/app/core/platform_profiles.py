"""Platform Delivery Profiles: capa de destino sobre los formatos geométricos.

FORMAT (9x16 / 1x1 / 16x9) = geometría del archivo de salida.
DESTINATION PROFILE = destino y sus reglas (loudness, codecs, safe zones,
duración máxima, color). Son conceptos ortogonales: 9x16 + tiktok y
9x16 + instagram_reels comparten renderer y difieren en reglas.

Single source of truth: render, delivery QC, captions y safe zones consumen
estos valores desde aquí; nada los duplica.

Compatibilidad legacy: los jobs existentes que solo envían formato se mapean
internamente a generic_vertical / generic_square / generic_landscape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformProfile:
    id: str
    display_name: str
    # Geometría de referencia del destino (la del render la fija el formato).
    frame_width: int
    frame_height: int
    aspect: str
    fps_preferred: float
    fps_max: float
    allowed_codecs: tuple[str, ...]
    preferred_codec: str
    pixel_format: str
    audio_codec: str
    sample_rate: int
    channels: int
    target_lufs: float
    true_peak: float
    max_duration_seconds: float | None
    allow_hdr: bool
    prefer_sdr: bool
    safe_top: float
    safe_bottom: float
    safe_left: float
    safe_right: float
    captions: dict = field(default_factory=dict)
    # Tolerancias de calidad (QC).
    loudness_warn_lu: float = 1.5
    loudness_fail_lu: float = 4.0
    true_peak_warn_db: float = 0.5
    true_peak_fail_db: float = 1.0
    fps_tolerance: float = 0.5
    allow_vfr: bool = False
    requires_faststart: bool = True

    def to_dict(self) -> dict:
        """Contrato JSON anidado para API/frontend."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "frame": {
                "width": self.frame_width,
                "height": self.frame_height,
                "aspect": self.aspect,
            },
            "fps": {"preferred": self.fps_preferred, "max": self.fps_max},
            "video": {
                "allowed_codecs": list(self.allowed_codecs),
                "preferred_codec": self.preferred_codec,
                "pixel_format": self.pixel_format,
            },
            "audio": {
                "preferred_codec": self.audio_codec,
                "target_lufs": self.target_lufs,
                "true_peak": self.true_peak,
                "channels": self.channels,
                "sample_rate": self.sample_rate,
            },
            "duration": {"max_seconds": self.max_duration_seconds},
            "color": {"allow_hdr": self.allow_hdr, "prefer_sdr": self.prefer_sdr},
            "safe_zones": {
                "top": self.safe_top,
                "bottom": self.safe_bottom,
                "left": self.safe_left,
                "right": self.safe_right,
            },
            "captions": dict(self.captions),
            "quality": {
                "loudness_warn_lu": self.loudness_warn_lu,
                "loudness_fail_lu": self.loudness_fail_lu,
                "true_peak_warn_db": self.true_peak_warn_db,
                "true_peak_fail_db": self.true_peak_fail_db,
                "fps_tolerance": self.fps_tolerance,
                "allow_vfr": self.allow_vfr,
                "requires_faststart": self.requires_faststart,
            },
        }


def _vertical(display_name: str, profile_id: str, **overrides) -> PlatformProfile:
    base = dict(
        id=profile_id,
        display_name=display_name,
        frame_width=1080,
        frame_height=1920,
        aspect="9:16",
        fps_preferred=30.0,
        fps_max=60.0,
        allowed_codecs=("h264", "hevc"),
        preferred_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
        sample_rate=44100,
        channels=2,
        target_lufs=-14.0,
        true_peak=-1.5,
        max_duration_seconds=None,
        allow_hdr=False,
        prefer_sdr=True,
        safe_top=0.05,
        safe_bottom=0.10,
        safe_left=0.05,
        safe_right=0.05,
        captions={"default_style": "viral_bold", "karaoke": True},
    )
    base.update(overrides)
    return PlatformProfile(**base)


PROFILES: dict[str, PlatformProfile] = {
    p.id: p
    for p in (
        # --- Genéricos (compatibilidad legacy y destino neutro) ---
        _vertical("Vertical (genérico)", "generic_vertical"),
        PlatformProfile(
            id="generic_square",
            display_name="Cuadrado (genérico)",
            frame_width=1080,
            frame_height=1080,
            aspect="1:1",
            fps_preferred=30.0,
            fps_max=60.0,
            allowed_codecs=("h264", "hevc"),
            preferred_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
            target_lufs=-14.0,
            true_peak=-1.5,
            max_duration_seconds=None,
            allow_hdr=False,
            prefer_sdr=True,
            safe_top=0.05,
            safe_bottom=0.10,
            safe_left=0.05,
            safe_right=0.05,
        ),
        PlatformProfile(
            id="generic_landscape",
            display_name="Horizontal (genérico)",
            frame_width=1920,
            frame_height=1080,
            aspect="16:9",
            fps_preferred=30.0,
            fps_max=60.0,
            allowed_codecs=("h264", "hevc"),
            preferred_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
            target_lufs=-14.0,
            true_peak=-1.5,
            max_duration_seconds=None,
            allow_hdr=False,
            prefer_sdr=True,
            safe_top=0.05,
            safe_bottom=0.08,
            safe_left=0.05,
            safe_right=0.05,
        ),
        # --- Plataformas ---
        _vertical(
            "Instagram Reels",
            "instagram_reels",
            max_duration_seconds=180.0,
            true_peak=-1.0,
        ),
        PlatformProfile(
            id="instagram_post",
            display_name="Instagram (post)",
            frame_width=1080,
            frame_height=1350,
            aspect="4:5",
            fps_preferred=30.0,
            fps_max=30.0,
            allowed_codecs=("h264",),
            preferred_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
            target_lufs=-14.0,
            true_peak=-1.0,
            max_duration_seconds=60.0,
            allow_hdr=False,
            prefer_sdr=True,
            safe_top=0.05,
            safe_bottom=0.10,
            safe_left=0.06,
            safe_right=0.06,
        ),
        _vertical(
            "TikTok",
            "tiktok",
            max_duration_seconds=600.0,
            safe_top=0.08,
            safe_bottom=0.15,
            safe_right=0.10,
        ),
        _vertical("YouTube Shorts", "youtube_shorts", max_duration_seconds=180.0),
        PlatformProfile(
            id="youtube",
            display_name="YouTube",
            frame_width=1920,
            frame_height=1080,
            aspect="16:9",
            fps_preferred=30.0,
            fps_max=60.0,
            allowed_codecs=("h264", "hevc", "vp9", "av1"),
            preferred_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            sample_rate=48000,
            channels=2,
            target_lufs=-14.0,
            true_peak=-1.0,
            max_duration_seconds=None,
            allow_hdr=True,
            prefer_sdr=False,
            safe_top=0.05,
            safe_bottom=0.08,
            safe_left=0.05,
            safe_right=0.05,
        ),
        _vertical("Facebook Reels", "facebook_reels", max_duration_seconds=90.0),
        PlatformProfile(
            id="x_video",
            display_name="X (video)",
            frame_width=1280,
            frame_height=720,
            aspect="16:9",
            fps_preferred=30.0,
            fps_max=60.0,
            allowed_codecs=("h264",),
            preferred_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
            target_lufs=-14.0,
            true_peak=-1.0,
            max_duration_seconds=140.0,
            allow_hdr=False,
            prefer_sdr=True,
            safe_top=0.05,
            safe_bottom=0.08,
            safe_left=0.05,
            safe_right=0.05,
        ),
        PlatformProfile(
            id="linkedin_video",
            display_name="LinkedIn",
            frame_width=1920,
            frame_height=1080,
            aspect="16:9",
            fps_preferred=30.0,
            fps_max=30.0,
            allowed_codecs=("h264",),
            preferred_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
            target_lufs=-14.0,
            true_peak=-1.0,
            max_duration_seconds=600.0,
            allow_hdr=False,
            prefer_sdr=True,
            safe_top=0.05,
            safe_bottom=0.10,
            safe_left=0.05,
            safe_right=0.05,
        ),
    )
}


# Mapeo legacy: los jobs que solo envían formato siguen funcionando igual.
LEGACY_FORMAT_MAP = {
    "9x16": "generic_vertical",
    "1x1": "generic_square",
    "16x9": "generic_landscape",
}


def profile_for_format(fmt: str) -> PlatformProfile:
    """Perfil genérico correspondiente a un formato geométrico."""
    return PROFILES[LEGACY_FORMAT_MAP.get(fmt, "generic_vertical")]


def resolve_profile(
    profile_id: str | None,
    fmt: str | None = None,
) -> PlatformProfile:
    """Resuelve el perfil efectivo.

    - id válido conocido → ese perfil.
    - None / '' / 'auto' / desconocido → genérico según el formato (legacy).
    """
    if profile_id:
        normalized = str(profile_id).strip().lower()
        if normalized in PROFILES:
            return PROFILES[normalized]
    return profile_for_format(fmt or "9x16")


def catalog() -> list[dict]:
    """Catálogo compacto para selects de UI (sin reglas internas)."""
    return [
        {
            "id": profile.id,
            "display_name": profile.display_name,
            "aspect": profile.aspect,
            "frame": f"{profile.frame_width}x{profile.frame_height}",
            "max_seconds": profile.max_duration_seconds,
        }
        for profile in PROFILES.values()
    ]
