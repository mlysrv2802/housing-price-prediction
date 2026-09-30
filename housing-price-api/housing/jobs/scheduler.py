"""
scheduler.py
-------------
The Python equivalent of Spring's @Scheduled -- runs jobs on a timer,
inside a single long-lived process. Conceptually:

    Spring:  @Scheduled(fixedRate = 3600000)  // every hour, milliseconds
    Python:  scheduler.add_job(job, 'interval', hours=1)

This closes the loop you asked about earlier: fetch new data -> validate ->
append -> IF new data actually arrived -> retrain -> compare against the
previous model's metrics -> only keep the new model if it's not worse.

This process needs to stay running (like a Spring Boot app does) --
it's not a one-shot script. In production you'd run this as a background
service (systemd on Linux, NSSM/Task Scheduler on Windows, or inside a
Docker container), not by leaving a terminal window open.

Run: python -m housing.jobs.scheduler
Stop: Ctrl+C
"""

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from housing import config
from housing.data import fetch_new_data
from housing.ml import model_registry
from housing.ml import train
from housing.data import generate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_if_needed() -> None:
    """Ensures a production model exists BEFORE anything tries to reach the
    API. Breaks the circular dependency: the API container needs a model
    to start; the old logic only trained AFTER a successful fetch from
    that same (not-yet-running) API. This runs independent of the API
    entirely, using whatever data is already on disk (or generating some)."""
    if config.PRODUCTION_MODEL_PATH.exists():
        logger.info("Production model already exists -- skipping bootstrap.")
        return

    logger.info("No production model found -- bootstrapping one now (API not required for this step).")

    if not config.RAW_DATA_PATH.exists():
        logger.info("No training data found either -- generating some first.")
        generate_data.main()

    train.main()
    logger.info("Bootstrap complete -- production model now exists.")

def fetch_and_retrain_job() -> None:
    """The actual scheduled job -- this is the function that runs on
    every tick of the timer, same role as a Spring @Scheduled method body."""
    logger.info("=== Scheduled job starting: fetch + conditional retrain ===")

    try:
        rows_added = fetch_new_data.main()
    except SystemExit:
        # fetch_new_data.main() calls sys.exit(1) on a hard failure (API down,
        # timeout). Catch it here so ONE bad run doesn't kill the whole
        # scheduler process -- the next scheduled tick should still fire.
        logger.error("Fetch step failed this run -- will retry on next scheduled tick.")
        return

    if not rows_added:
        logger.info("No new data this run -- skipping retrain.")
        return

    logger.info("%d new rows added -- triggering retrain.", rows_added)

    # train.py now handles the promote-vs-reject decision itself via
    # model_registry, and tells us directly what happened -- no more
    # guessing the outcome by comparing metric values (which breaks when
    # two runs happen to score identically).
    previous_metrics = model_registry.get_production_metrics()
    promoted = train.main()
    new_metrics = model_registry.get_production_metrics()

    if not promoted:
        logger.info(
            "Retrain ran, but production model is UNCHANGED (candidate was "
            "rejected by model_registry -- see the WARNING above for why)."
        )
    elif previous_metrics:
        logger.info(
            "Production model UPDATED. R^2: %.4f -> %.4f (%+.4f)",
            previous_metrics["r2_score"], new_metrics["r2_score"],
            new_metrics["r2_score"] - previous_metrics["r2_score"],
        )
    else:
        logger.info("First production model created. R^2: %.4f", new_metrics["r2_score"])


if __name__ == "__main__":
    bootstrap_if_needed()  
    
    scheduler = BlockingScheduler()

    # Every N minutes for demo purposes -- in a real system this might be
    # hourly, nightly, or triggered by an event instead of a fixed interval.
    INTERVAL_MINUTES = 30

    # IMPORTANT: next_run_time=None does NOT mean "wait for the first
    # interval" -- it means "add this job PAUSED, with no run scheduled at
    # all" (a genuine APScheduler gotcha). To delay the first automatic
    # fire until one interval from now, pass an explicit datetime instead.
    scheduler.add_job(
        fetch_and_retrain_job,
        trigger=IntervalTrigger(minutes=INTERVAL_MINUTES),
        id="fetch_and_retrain",
        next_run_time=datetime.now() + timedelta(minutes=INTERVAL_MINUTES),
    )

    logger.info("Scheduler started. Job will run every %d minutes. Press Ctrl+C to stop.", INTERVAL_MINUTES)
    logger.info("Running one job immediately so you can see it work right now...")
    fetch_and_retrain_job()  # run once immediately for demo visibility

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
