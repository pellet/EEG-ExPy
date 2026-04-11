"""Normative P100 latency model for the Pattern Reversal VEP experiment.

STATUS: spike / proof-of-concept — NOT USABLE FOR REAL ANALYSIS YET.

This module is the target of the ``spike/vep_normative_regression`` branch.
Its job is to replace the binned-table normative approach (see
``doc/experiments/vvep.rst`` Plan B section on the MVP branch) with a
continuous regression model of P100 latency on head size, age, and sex.

The intended API shape::

    from eegnb.analysis.vep_norms import fit_normative_model, NormativeModel

    # One-off: fit from your in-lab baseline cohort
    model = fit_normative_model(baseline_df)          # -> NormativeModel
    model.save("muse2_vr_60hz_norms.json")

    # Per new participant:
    model = NormativeModel.load("muse2_vr_60hz_norms.json")
    predicted_ms = model.predict(
        head_circumference_mm=570,
        age_years=34,
        sex="F",
    )
    z = model.z_score(
        measured_latency_ms=104.2,
        head_circumference_mm=570,
        age_years=34,
        sex="F",
    )

Open questions before this can leave the spike branch:

1. Functional form. Linear is the obvious first pass but scalp volume
   scales with head radius cubed and conduction velocity varies along
   the visual pathway, so log or square-root terms may fit better. Fit
   both and compare residuals on a real cohort before committing.
2. Which head-size metric is primary? Head circumference is easiest and
   most commonly recorded, but nasion-to-inion is the closest proxy for
   visual pathway length. The API accepts all three and picks whichever
   is available, but the fit code needs to decide which to weight most.
3. How to handle missing metadata. In practice participants will often
   have head circumference but not nasion-to-inion. The fit needs to
   handle this gracefully (multiple imputation, or just fitting whichever
   columns are complete and falling back appropriately).
4. Interocular-difference model. The interocular P100 latency difference
   cancels out fixed offsets and head-size effects exactly in the
   subtraction, so it needs a different model (or just a simple mean/SD
   from baseline, since there's nothing to regress against). Decide
   whether to include that here or in a sibling module.
5. Literature coefficients. The placeholder values in
   ``_PLACEHOLDER_COEFFS`` below are not real and should never be used
   for inference. A literature pass is needed before anyone relies on
   a default-fit model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json


# Placeholder coefficients — order-of-magnitude guesses from the literature
# so the API is testable end-to-end. DO NOT USE FOR REAL ANALYSIS.
_PLACEHOLDER_COEFFS = {
    "intercept_ms": 80.0,
    "head_size_ms_per_mm": 0.15,
    "age_ms_per_year": 0.05,
    "sex_ms_female_offset": -1.0,
    "residual_sd_ms": 5.0,
}


Sex = Literal["M", "F", "other"]


@dataclass
class NormativeModel:
    """Continuous normative model for P100 latency.

    Attributes are the fitted coefficients of::

        P100_latency_ms = intercept
                        + head_size_coef * head_size_mm
                        + age_coef       * age_years
                        + sex_coef       * (1 if sex == "F" else 0)
                        + ε,    ε ~ N(0, residual_sd^2)

    The head-size metric is whichever of head_circumference_mm,
    interaural_arc_mm, or nasion_inion_mm was used to fit the model,
    recorded in ``head_size_metric``.
    """
    intercept_ms: float
    head_size_coef_ms_per_mm: float
    age_coef_ms_per_year: float
    sex_coef_ms_female: float
    residual_sd_ms: float
    head_size_metric: Literal[
        "head_circumference_mm", "interaural_arc_mm", "nasion_inion_mm"
    ]
    n_baseline_participants: int
    notes: str = ""

    def predict(
        self,
        head_size_mm: float,
        age_years: float,
        sex: Sex,
    ) -> float:
        """Predicted P100 latency (ms) for the given participant."""
        female = 1.0 if sex == "F" else 0.0
        return (
            self.intercept_ms
            + self.head_size_coef_ms_per_mm * head_size_mm
            + self.age_coef_ms_per_year * age_years
            + self.sex_coef_ms_female * female
        )

    def z_score(
        self,
        measured_latency_ms: float,
        head_size_mm: float,
        age_years: float,
        sex: Sex,
    ) -> float:
        """How many residual SDs from prediction is the measured latency?"""
        predicted = self.predict(head_size_mm, age_years, sex)
        return (measured_latency_ms - predicted) / self.residual_sd_ms

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormativeModel":
        with open(path) as f:
            return cls(**json.load(f))


def fit_normative_model(
    baseline_df,
    head_size_metric: str = "head_circumference_mm",
) -> NormativeModel:
    """Fit a NormativeModel to a baseline cohort.

    SPIKE STATUS: this currently returns placeholder coefficients, not a
    real fit. Replacing the body with a real statsmodels / sklearn OLS
    fit is the main todo before this can leave the spike branch.

    ``baseline_df`` is expected to be a pandas DataFrame with columns::

        p100_latency_ms    float
        head_circumference_mm  (or interaural_arc_mm / nasion_inion_mm)  float
        age_years          float
        sex                str in {"M", "F", "other"}

    One row per baseline participant (averaged across eyes, or per-eye
    with a participant-level random effect — decision deferred).
    """
    raise NotImplementedError(
        "Real fit not yet implemented — see module docstring for the list "
        "of decisions that need to be made first. Use "
        "`placeholder_model()` for API testing only."
    )


def placeholder_model(
    head_size_metric: str = "head_circumference_mm",
) -> NormativeModel:
    """Return a NormativeModel with placeholder coefficients.

    For API / integration testing only. The coefficients are
    order-of-magnitude guesses and must not be used for real inference.
    """
    return NormativeModel(
        intercept_ms=_PLACEHOLDER_COEFFS["intercept_ms"],
        head_size_coef_ms_per_mm=_PLACEHOLDER_COEFFS["head_size_ms_per_mm"],
        age_coef_ms_per_year=_PLACEHOLDER_COEFFS["age_ms_per_year"],
        sex_coef_ms_female=_PLACEHOLDER_COEFFS["sex_ms_female_offset"],
        residual_sd_ms=_PLACEHOLDER_COEFFS["residual_sd_ms"],
        head_size_metric=head_size_metric,
        n_baseline_participants=0,
        notes="PLACEHOLDER — not fit on any real data.",
    )
