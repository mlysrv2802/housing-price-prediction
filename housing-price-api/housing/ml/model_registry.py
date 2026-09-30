"""
model_registry.py
--------------------
Handles model VERSIONING, PROMOTION, and ROLLBACK -- the missing piece
flagged earlier: a worse retrain should never be able to silently replace
a working production model.

Mental model (same idea as a deployment pipeline):
  train.py produces a CANDIDATE  ->  save_version() files it away, untouched
  promote_if_better() checks it against the CURRENT PRODUCTION metrics
  -> if it's not meaningfully worse: promote it (becomes the new production model)
  -> if it IS meaningfully worse: reject it, log why, production is untouched

Every version is kept on disk under models/versions/, so you can always
manually roll back to an older one if something slips through.
"""

import json
import logging
import shutil
from datetime import datetime

import joblib

from housing import config

logger = logging.getLogger(__name__)


def save_version(pipeline, metrics: dict) -> str:
    """Saves a trained pipeline + its metrics as a new, permanent version.
    Returns the version_id (a timestamp string) -- never overwrites an
    existing version, so your full training history is always recoverable."""
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = config.VERSIONS_DIR / f"model_{version_id}.joblib"
    metrics_path = config.VERSIONS_DIR / f"metrics_{version_id}.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Saved new model version: %s", version_id)
    return version_id


def get_production_metrics() -> dict | None:
    """Returns the currently-live production model's metrics, or None if
    no model has ever been promoted yet (e.g. very first training run)."""
    if not config.PRODUCTION_METRICS_PATH.exists():
        return None
    return json.loads(config.PRODUCTION_METRICS_PATH.read_text())


def promote(version_id: str) -> None:
    """Copies a specific version's artifacts to the production path.
    Used both by the automatic promotion check AND for manual rollback --
    promoting an OLD version is exactly what a rollback is."""
    model_path = config.VERSIONS_DIR / f"model_{version_id}.joblib"
    metrics_path = config.VERSIONS_DIR / f"metrics_{version_id}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No such version: {version_id} (looked for {model_path})")

    shutil.copy(model_path, config.PRODUCTION_MODEL_PATH)
    shutil.copy(metrics_path, config.PRODUCTION_METRICS_PATH)
    logger.info("Promoted version %s to production.", version_id)


def promote_if_better(version_id: str, candidate_metrics: dict) -> bool:
    """The actual gate. Returns True if promoted, False if rejected.

    Promotion rule: promote if there's no production model yet, OR if the
    candidate's R^2 hasn't dropped by more than config.MAX_ACCEPTABLE_R2_DROP
    versus the current production model. This is intentionally simple --
    real systems often add more checks (RMSE, latency, fairness metrics),
    but the GATING PATTERN is the same.
    """
    production_metrics = get_production_metrics()

    if production_metrics is None:
        logger.info("No existing production model -- promoting %s as the first one.", version_id)
        promote(version_id)
        return True

    r2_change = candidate_metrics["r2_score"] - production_metrics["r2_score"]

    if r2_change < -config.MAX_ACCEPTABLE_R2_DROP:
        logger.warning(
            "REJECTED version %s: R^2 would drop by %.4f (from %.4f to %.4f), "
            "exceeding the allowed threshold of %.4f. Production model unchanged.",
            version_id, -r2_change, production_metrics["r2_score"],
            candidate_metrics["r2_score"], config.MAX_ACCEPTABLE_R2_DROP,
        )
        return False

    logger.info(
        "Promoting version %s: R^2 %.4f -> %.4f (change: %+.4f)",
        version_id, production_metrics["r2_score"], candidate_metrics["r2_score"], r2_change,
    )
    promote(version_id)
    return True


def list_versions() -> list[dict]:
    """Lists every saved version with its metrics, newest first -- useful
    for deciding what to roll back to."""
    versions = []
    for metrics_file in sorted(config.VERSIONS_DIR.glob("metrics_*.json"), reverse=True):
        version_id = metrics_file.stem.replace("metrics_", "")
        metrics = json.loads(metrics_file.read_text())
        versions.append({"version_id": version_id, **metrics})
    return versions


def rollback_to(version_id: str) -> None:
    """Explicit manual rollback -- same mechanism as promote(), but named
    for intent: a human decided production should go BACKWARD."""
    logger.info("Rolling back production to version %s.", version_id)
    promote(version_id)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] not in {"list", "rollback"}:
        print("Usage:")
        print("  python -m housing.ml.model_registry list                 # show all versions")
        print("  python -m housing.ml.model_registry rollback <version_id> # promote an older version")
        sys.exit(1)

    if sys.argv[1] == "list":
        current = get_production_metrics()
        print(f"Current production metrics: {current}\n")
        for v in list_versions():
            marker = ""
            print(f"  {v['version_id']}  |  R^2={v['r2_score']:.4f}  RMSE={v['rmse']:,.0f}{marker}")

    elif sys.argv[1] == "rollback":
        if len(sys.argv) < 3:
            print("Please provide a version_id, e.g.:")
            print("  python -m housing.ml.model_registry rollback 20260921_192455")
            sys.exit(1)
        rollback_to(sys.argv[2])
        print(f"Rolled back to version {sys.argv[2]}. This is now the production model.")
