def test_metric(metric):
    tiny_metric, george_metric_args = metric
    george_metric = george.metrics.Metric(**george_metric_args)
    for n in range(george_metric.ndim):
        e = np.zeros(george_metric.ndim)
        e[n] = 1.0
        np.testing.assert_allclose(
            tiny_metric(e), e.T @ np.linalg.solve(george_metric.to_matrix(), e)
        )
