"""
Generates a synthetic but realistic dataset linking social media usage /
lifestyle habits to a student's mental health score (0-100).

The ground-truth relationship is hand-designed to reflect well-documented
patterns from mental health research (e.g. excessive screen time before bed,
social comparison, and cyberbullying hurting wellbeing; sleep, physical
activity, and face-to-face interaction helping it) plus random noise, so the
downstream ML model has real signal to learn instead of pure randomness.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3000


def generate_dataset(n=N):
    age = RNG.integers(13, 25, n)  # school + college range
    gender = RNG.choice(["Male", "Female", "Other"], n, p=[0.47, 0.47, 0.06])

    daily_social_media_hours = np.clip(RNG.normal(3.5, 1.8, n), 0, 14)
    num_platforms_used = np.clip(RNG.poisson(3, n), 0, 9)
    sleep_hours = np.clip(RNG.normal(6.8, 1.3, n), 2, 11)
    physical_activity_hours = np.clip(RNG.exponential(0.8, n), 0, 5)
    screen_time_before_bed_min = np.clip(RNG.normal(45, 30, n), 0, 180)
    study_hours = np.clip(RNG.normal(3, 1.5, n), 0, 10)
    face_to_face_hours = np.clip(RNG.exponential(1.5, n), 0, 8)
    academic_performance = np.clip(RNG.normal(72, 13, n), 30, 100)  # % marks
    social_comparison_score = np.clip(RNG.normal(5, 2.2, n), 1, 10)  # 1-10 self rated
    cyberbullying_experience = RNG.choice([0, 1], n, p=[0.82, 0.18])

    # ---- Ground truth formula for mental_health_score (0-100, higher=better) ----
    score = 70.0

    # Excess social media use hurts, mild use is fine (inflection ~2.5 hrs)
    score += -2.6 * np.clip(daily_social_media_hours - 2.5, 0, None)

    # Too many platforms = more fragmented attention / comparison exposure
    score += -0.8 * np.clip(num_platforms_used - 3, 0, None)

    # Sleep: below 7 hours hurts a lot, above 9 has diminishing returns
    score += 3.2 * np.clip(sleep_hours - 6, -4, 3)

    # Physical activity helps
    score += 3.0 * np.clip(physical_activity_hours, 0, 3)

    # Screen time right before bed hurts (poor sleep quality, blue light etc.)
    score += -0.045 * screen_time_before_bed_min

    # Healthy study routine has a small positive effect, extreme overload hurts
    score += 1.1 * np.clip(study_hours, 0, 4) - 0.6 * np.clip(study_hours - 6, 0, None)

    # Face-to-face social interaction strongly protective
    score += 2.3 * np.clip(face_to_face_hours, 0, 4)

    # Academic performance has a mild positive relationship
    score += 0.06 * (academic_performance - 70)

    # Social comparison (doom-scrolling, comparing life to influencers) hurts
    score += -1.8 * (social_comparison_score - 5)

    # Cyberbullying has a large negative effect
    score += -16.0 * cyberbullying_experience

    # Random individual variation / unobserved factors
    score += RNG.normal(0, 5.0, n)

    mental_health_score = np.clip(score, 0, 100).round(1)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "daily_social_media_hours": daily_social_media_hours.round(2),
        "num_platforms_used": num_platforms_used,
        "sleep_hours": sleep_hours.round(2),
        "physical_activity_hours": physical_activity_hours.round(2),
        "screen_time_before_bed_min": screen_time_before_bed_min.round(1),
        "study_hours": study_hours.round(2),
        "face_to_face_hours": face_to_face_hours.round(2),
        "academic_performance": academic_performance.round(1),
        "social_comparison_score": social_comparison_score.round(1),
        "cyberbullying_experience": cyberbullying_experience,
        "mental_health_score": mental_health_score,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("student_mental_health_data.csv", index=False)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nTarget stats:\n", df["mental_health_score"].describe())
