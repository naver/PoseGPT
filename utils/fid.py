import numpy as np
from scipy import linalg
import torch

# Adapted from https://github.com/mseitzer/pytorch-fid

def calculate_activation_statistics(activations):
    """ Compute mean and covariance of activations"""
    activations = activations.cpu().numpy()
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet distance between two multivariate Gaussians
    X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2):
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    print("Computing FID ...", end='')
    mu1, mu2, sigma1, sigma2 = [np.atleast_1d(x) for x in [mu1, mu2, sigma1, sigma2]]
    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape, 'Incoherent vector shapes'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    print('OK!')
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])

def calculate_diversity_multimodality(feats, labels, nb_runs=10, nb_classes=12):
    """ Runs _calculate_diversity_multimodality [nb_runs] times, to have more reliable estimates. """
    div_mod = list(zip(*[_calculate_diversity_multimodality(feats, labels, nb_classes) for _ in range(nb_runs)]))
    return np.asarray(div_mod[0]).mean(), np.asarray(div_mod[1]).mean()

def multiclass_div_mod(activations, labels, nb_classes):
    """ When a sequence has multiple labels, we copy it once for each and evaluate them separately.
    """
    num_classes = labels[0].shape[0]
    act_and_lab = zip(torch.split(activations, 1, dim=0), labels)
    activations, labels = [], []
    for a, l in act_and_lab:
        ll = torch.split(torch.where(l)[0], 1, dim=0)
        activations.extend([a for _l in ll])
        labels.extend([_l.item() for _l in ll])
    labels = torch.Tensor(labels)
    return calculate_diversity_multimodality(torch.cat(activations), labels, nb_classes=nb_classes)
    

def calculate_diversity(activations, labels):
    diversity_times = 200
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    # NOTE this is pretty dumb. Also, why is not vectorized? 
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    return diversity.item()

def calculate_multimodality(activations, labels, num_labels):
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    multimodality = 0
    label_quotas = np.repeat(multimodality_times, num_labels)
    i, max_run = 0, 1000 # With long tailed class distribution, the loop can be very long

    # TODO @tlucas: Change the sampling procedure to be robust to imbalanced classes.
    while np.any(label_quotas > 0) and i < max_run:
        # print(label_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        i += 1
        if not label_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        label_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)

    multimodality /= (multimodality_times * num_labels)
    return multimodality.item()

def _calculate_diversity_multimodality(activations, labels, num_labels):
    return calculate_diversity(activations, labels), calculate_multimodality(activations, labels, num_labels)

