from typing import List

import pytest
import numpy as np
import pandas as pd

from obp.dataset import SyntheticSlateBanditDataset
from obp.dataset.synthetic_slate import (
    logistic_weighted_reward_function,
    linear_behavior_policy_logit,
)
from obp.types import BanditFeedback


# n_actions, len_list, dim_context, reward_type, random_state, description
invalid_input_of_init = [
    ("4", 3, 2, "binary", 1, "n_actions must be an integer larger than 1"),
    (1, 3, 2, "binary", 1, "n_actions must be an integer larger than 1"),
    (5, "4", 2, "binary", 1, "len_list must be an integer such that"),
    (5, -1, 2, "binary", 1, "len_list must be an integer such that"),
    (5, 10, 2, "binary", 1, "len_list must be an integer such that"),
    (5, 3, 0, "binary", 1, "dim_context must be a positive integer"),
    (5, 3, "2", "binary", 1, "dim_context must be a positive integer"),
    (5, 3, 2, "aaa", 1, "reward_type must be either"),
    (5, 3, 2, "binary", "x", "random_state must be an integer"),
    (5, 3, 2, "binary", None, "random_state must be an integer"),
]


@pytest.mark.parametrize(
    "n_actions, len_list, dim_context, reward_type, random_state, description",
    invalid_input_of_init,
)
def test_synthetic_slate_init_using_invalid_inputs(
    n_actions, len_list, dim_context, reward_type, random_state, description
):
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = SyntheticSlateBanditDataset(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=dim_context,
            reward_type=reward_type,
            random_state=random_state,
        )


def check_slate_bandit_feedback(bandit_feedback: BanditFeedback):
    # check pscore columns
    pscore_columns: List[str] = []
    pscore_candidate_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    for column in pscore_candidate_columns:
        if column in bandit_feedback and bandit_feedback[column] is not None:
            pscore_columns.append(column)
        else:
            pscore_columns.append(column)
    assert (
        len(pscore_columns) > 0
    ), f"bandit feedback must contains at least one of the following pscore columns: {pscore_candidate_columns}"
    bandit_feedback_df = pd.DataFrame()
    for column in ["impression_id", "position", "action"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    # sort dataframe
    bandit_feedback_df = (
        bandit_feedback_df.sort_values(["impression_id", "position"])
        .reset_index(drop=True)
        .copy()
    )
    # check uniqueness
    assert (
        bandit_feedback_df.duplicated(["impression_id", "position"]).sum() == 0
    ), "position must not be duplicated in each impression"
    assert (
        bandit_feedback_df.duplicated(["impression_id", "action"]).sum() == 0
    ), "action must not be duplicated in each impression"
    # check pscores
    for column in pscore_columns:
        invalid_pscore_flgs = (bandit_feedback_df[column] < 0) | (
            bandit_feedback_df[column] > 1
        )
        assert invalid_pscore_flgs.sum() == 0, "the range of pscores must be [0, 1]"
    if "pscore_joint_above" in pscore_columns and "pscore_joint_all" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_joint_above"]
            < bandit_feedback_df["pscore_joint_all"]
        ).sum() == 0, "pscore_joint_above is smaller or equal to pscore_joint_all"
    if "pscore_marginal" in pscore_columns and "pscore_joint_all" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_marginal"]
            < bandit_feedback_df["pscore_joint_all"]
        ).sum() == 0, "pscore_joint_all is smaller or equal to pscore_marginal"
    if "pscore_marginal" in pscore_columns and "pscore_joint_above" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_marginal"]
            < bandit_feedback_df["pscore_joint_above"]
        ).sum() == 0, "pscore_joint_above is smaller or equal to pscore_marginal"
    if "pscore_joint_above" in pscore_columns:
        previous_minimum_pscore_joint_above = (
            bandit_feedback_df.groupby("impression_id")["pscore_joint_above"]
            .expanding()
            .min()
            .values
        )
        assert (
            previous_minimum_pscore_joint_above
            < bandit_feedback_df["pscore_joint_above"]
        ).sum() == 0, (
            "pscore_joint_above must be non-decresing sequence in each impression"
        )
    if "pscore_joint_all" in pscore_columns:
        count_pscore_joint_all_in_expression = bandit_feedback_df.groupby(
            "impression_id"
        ).apply(lambda x: x["pscore_joint_all"].unique().shape[0])
        assert (
            count_pscore_joint_all_in_expression != 1
        ).sum() == 0, "pscore_joint_all must be unique in each impression"
    if "pscore_joint_all" in pscore_columns and "pscore_joint_above" in pscore_columns:
        last_slot_feedback_df = bandit_feedback_df.drop_duplicates(
            "impression_id", keep="last"
        )
        assert (
            last_slot_feedback_df["pscore_joint_all"]
            != last_slot_feedback_df["pscore_joint_above"]
        ).sum() == 0, (
            "pscore_joint_all must be the same as pscore_joint_above in the last slot"
        )


def test_synthetic_slate_obtain_batch_bandit_feedback_using_uniform_random_behavior_policy():
    # set parameters
    n_actions = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 100
    dataset = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["impression_id", "position", "action"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    # check pscore marginal
    pscore_marginal = float(len_list / n_actions)
    assert np.allclose(
        bandit_feedback_df["pscore_marginal"].unique(), [pscore_marginal]
    ), f"pscore_marginal must be [{pscore_marginal}], but {bandit_feedback_df['pscore_marginal'].unique()}"
    # check pscore joint
    pscore_joint_above = []
    pscore_above = 1.0
    for position_ in range(len_list):
        pscore_above = pscore_above * 1.0 / (n_actions - position_)
        pscore_joint_above.append(pscore_above)
    assert np.allclose(
        bandit_feedback_df["pscore_joint_above"], np.tile(pscore_joint_above, n_rounds)
    ), f"pscore_joint_above must be {pscore_joint_above} for all impresessions"
    assert np.allclose(
        bandit_feedback_df["pscore_joint_all"].unique(), [pscore_above]
    ), f"pscore_joint_all must be {pscore_above} for all impressions"


def test_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy():
    # set parameters
    n_actions = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 100
    dataset = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    # print reward
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["impression_id", "position", "action", "reward"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    print(bandit_feedback_df.groupby("position")["reward"].describe())


def test_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy_without_pscore_marginal():
    # set parameters
    n_actions = 80
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 100
    dataset = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_marginal=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    assert (
        bandit_feedback["pscore_marginal"] is None
    ), f"pscore marginal must be None, but {bandit_feedback['pscore_marginal']}"


def test_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy_and_sips_logistic_reward():
    # set parameters
    n_actions = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 100
    reward_structure = "SIPS"
    dataset = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        exam_weight=1 / np.exp(np.arange(len_list)),
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_function=logistic_weighted_reward_function,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_marginal=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["impression_id", "position", "action", "reward"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    print(bandit_feedback_df.groupby("position")["reward"].describe())


def test_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy_and_rips_logistic_reward():
    # set parameters
    n_actions = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 100
    reward_structure = "RIPS"
    dataset = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        exam_weight=1 / np.exp(np.arange(len_list)),
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_function=logistic_weighted_reward_function,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_marginal=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["impression_id", "position", "action", "reward"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    print(bandit_feedback_df.groupby("position")["reward"].describe())


def test_tmp_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy_and_rips_logistic_reward():
    # set parameters
    n_actions = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 123
    n_rounds = 10000
    dataset_r = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure="RIPS",
        exam_weight=1 / np.exp(np.arange(len_list)),
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_function=logistic_weighted_reward_function,
    )
    # get feedback
    bandit_feedback_r = dataset_r.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_marginal=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback_r)
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df_r = pd.DataFrame()
    for column in [
        "impression_id",
        "position",
        "action",
        "reward",
        "expected_reward_factual",
    ] + pscore_columns:
        bandit_feedback_df_r[column] = bandit_feedback_r[column]
    print(bandit_feedback_df_r.groupby("position")["reward"].describe())
    # sips
    dataset_s = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure="SIPS",
        exam_weight=1 / np.exp(np.arange(len_list)),
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_function=logistic_weighted_reward_function,
    )
    # get feedback
    bandit_feedback_s = dataset_s.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_marginal=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback_s)
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df_s = pd.DataFrame()
    for column in [
        "impression_id",
        "position",
        "action",
        "reward",
        "expected_reward_factual",
    ] + pscore_columns:
        bandit_feedback_df_s[column] = bandit_feedback_s[column]
    print(bandit_feedback_df_s.groupby("position")["reward"].describe())
    # iips
    dataset_i = SyntheticSlateBanditDataset(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure="IIPS",
        exam_weight=1 / np.exp(np.arange(len_list)),
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
        reward_function=logistic_weighted_reward_function,
    )
    # get feedback
    bandit_feedback_i = dataset_i.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_marginal=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback_i)
    pscore_columns = [
        "pscore_joint_above",
        "pscore_joint_all",
        "pscore_marginal",
    ]
    bandit_feedback_df_i = pd.DataFrame()
    for column in [
        "impression_id",
        "position",
        "action",
        "reward",
        "expected_reward_factual",
    ] + pscore_columns:
        bandit_feedback_df_i[column] = bandit_feedback_i[column]
    print(bandit_feedback_df_i.groupby("position")["reward"].describe())
    # import pdb

    # pdb.set_trace()
