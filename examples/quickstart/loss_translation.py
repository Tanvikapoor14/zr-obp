from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression
import sys
import numpy as np
import torch
from bayes_opt import BayesianOptimization

# import open bandit pipeline (obp)
import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_reward_function
)
from obp.policy import (
    IPWLearner,
    QLearner,
    NNPolicyLearner,
    Random
)


def generateDataset():
    """Function to generate the dataset"""

    dataset = SyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        beta=-2,  # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="binary",  # "binary" or "continuous"
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    return dataset


def generateModel(dataset, m, l, r, p, lr):

    nn_ipw = NNPolicyLearner(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        off_policy_objective="ipw",
        policy_reg_param=p,
        batch_size=64,
        learning_rate_init=lr,
        max_iter=m,
        random_state=r,
        loss_translation=l
    )
    return nn_ipw


def graph_values(pred_actions, action, pscore, rewards):
    """Returns the numerator, denominator and ratio of the SNIPS estimator"""

    idx_tensor = torch.arange(action.shape[0], dtype=torch.long)
    iw = pred_actions[idx_tensor, action] / pscore
    num = np.mean(iw * rewards)
    den = np.mean(iw)
    ratio = num / den
    return num, den, ratio


def predict_value_den_train(m, r, p, lr, dataset, l, bandit_feedback_train):
    """Returns the denominator of the SNIPS estimator, after training on the
    training data"""

    model = generateModel(dataset, m, l, r, p, lr)

    # train NNPolicyLearner on the training set of logged bandit data
    model.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )

    # obtains action choice probabilities for the train set
    action_dist_nn_ipw_train = model.predict_proba(
        context=bandit_feedback_train["context"]
    )

    pred_actions_train = action_dist_nn_ipw_train[:, :, 0]
    action_train = bandit_feedback_train["action"]
    pscore_train = bandit_feedback_train["pscore"]
    rewards_train = bandit_feedback_train["reward"]

    num_train, den_train, ratio_train = graph_values(pred_actions_train, action_train, pscore_train, rewards_train)
    return den_train


def nearestPoint(lo, hi, search):
    input_lo = predict_value_den_train(lo)
    input_hi = predict_value_den_train(hi)
    ans = lo + (hi - lo) * (search - input_lo) // (input_hi - input_lo)
    return ans


def interpolationSearch(m, r, p, lr, dataset, bandit_feedback_train, lo, hi, increment, search_value, epsilon):
    """Performs an interpolation search on different values of loss translation lambda,
    to get the value of the denominator closest to 1, for the SNIPS estimator of
    the training data"""

    while lo <= hi:

        pos = nearestPoint(lo, hi, search_value)
        input_pos = predict_value_den_train(m, r, p, lr, dataset, pos, bandit_feedback_train)
        print(input_pos)

        if abs(input_pos - search_value) <= epsilon:
            return pos

        elif input_pos > search_value:
            return interpolationSearch(m, r, p, lr, dataset, bandit_feedback_train,
                                       pos + increment, hi, increment, search_value, epsilon)

        else:
            return interpolationSearch(m, r, p, lr, dataset, bandit_feedback_train,
                                       lo, pos - increment, increment, search_value, epsilon)

    return -1


def binarySearch(m, r, p, lr, dataset, bandit_feedback_train, lo, hi, increment, search_value, epsilon):
    """Performs a binary search on different values of loss translation lambda,
    to get the value of the denominator closest to 1, for the SNIPS estimator of
    the training data"""

    while lo <= hi:

        pos = lo + ((hi - lo) / 2)
        input_pos = predict_value_den_train(m, r, p, lr, dataset, pos, bandit_feedback_train)
        print(input_pos)

        if abs(input_pos - search_value) <= epsilon:
            return pos

        elif input_pos > search_value:
            return binarySearch(m, r, p, lr, dataset, bandit_feedback_train, pos + increment,
                                hi, increment, search_value, epsilon)

        else:
            return binarySearch(m, r, p, lr, dataset, bandit_feedback_train, lo,
                                pos - increment, increment, search_value, epsilon)

    return -1


def predict_value_ratio_train(m, r, p, lr, dataset, l, bandit_feedback_train):
    """Returns the SNIPS estimator ratio of the training data"""

    model = generateModel(dataset, m, l, r, p, lr)

    # train NNPolicyLearner on the training set of logged bandit data
    model.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )

    # obtains action choice probabilities for the train set
    action_dist_nn_ipw_train = model.predict_proba(
        context=bandit_feedback_train["context"]
    )

    pred_actions_train = action_dist_nn_ipw_train[:, :, 0]
    action_train = bandit_feedback_train["action"]
    pscore_train = bandit_feedback_train["pscore"]
    rewards_train = bandit_feedback_train["reward"]

    num_train, den_train, ratio_train = graph_values(pred_actions_train, action_train, pscore_train, rewards_train)
    return ratio_train


def bayesOpt(black_box_function, pbounds, init_pnts, num_iter):
    """Function for bayes optimization"""

    # Create a BayesianOptimization optimizer,
    # and optimize the given black_box_function.
    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds, verbose=2,
                                     random_state=4)
    optimizer.maximize(init_points=init_pnts, n_iter=num_iter)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
    return (optimizer.max["params"], optimizer.max["target"])


def predict_value_ratio_val(m, r , p, lr, dataset, l, bandit_feedback_train, bandit_feedback_val):
    """Returns the SNIPS estimator ratio, after training on the 
    validation data"""

    model = generateModel(dataset, m, l, r, p, lr)

    # train NNPolicyLearner on the training set of logged bandit data
    model.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )

    # obtains action choice probabilities for the train set
    action_dist_nn_ipw_val = model.predict_proba(
        context=bandit_feedback_val["context"]
    )

    pred_actions_val = action_dist_nn_ipw_val[:, :, 0]
    action_val = bandit_feedback_val["action"]
    pscore_val = bandit_feedback_val["pscore"]
    rewards_val = bandit_feedback_val["reward"]

    num_val, den_val, ratio_val = graph_values(pred_actions_val, action_val, pscore_val, rewards_val)
    return ratio_val
