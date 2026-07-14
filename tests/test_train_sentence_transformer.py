import numpy as np

from resumeai_agent.models.train_sentence_transformer import build_pair_features


def test_build_pair_features_contains_difference_product_and_similarity() -> None:
    resume_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    job_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    features = build_pair_features(resume_embeddings, job_embeddings)

    assert features.shape == (2, 5)
    assert np.allclose(features[0], [0.0, 0.0, 1.0, 0.0, 1.0])
    assert np.allclose(features[1], [1.0, 1.0, 0.0, 0.0, 0.0])
