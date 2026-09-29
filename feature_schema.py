"""Shared feature names; importing a schema must not load a prediction model."""

ALL_FEATURE_NAMES = [
    "beh_view_cnt_7d", "beh_view_duration_decay", "beh_calculator_use_cnt",
    "beh_compare_cnt", "beh_revisit_gap_avg", "txn_history_product_types",
    "txn_redemption_freq", "txn_avg_holding_period", "txn_days_since_last_purchase",
    "profile_aum_tier", "profile_lifecycle_stage", "profile_risk_level",
    "interact_push_open_rate_30d", "interact_advisor_contact_freq", "interact_last_script_accepted"
]
