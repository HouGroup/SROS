from sros.calculation.SROS import SRO


def _fake_sro(alpha, alpha_lili):
    sro = SRO.__new__(SRO)
    sro.a = alpha
    sro.a_LiLi = alpha_lili
    sro.a_LiLi_dict = {}
    sro.alpha_fix = lambda: sro.a
    sro.alpha_LiLi = lambda: (sro.a_LiLi, {})
    sro.exchange = lambda *args, **kwargs: sro.a
    sro.exchange_LiLi = lambda *args, **kwargs: sro.a_LiLi
    return sro


def test_run_does_not_report_both_reached_when_alpha_target_is_missed(capsys):
    sro = _fake_sro(alpha=-0.047, alpha_lili=0.0)

    status = sro.run(
        max_steps=3,
        target_alpha=-0.2,
        target_alpha_LiLi=0.0,
        random_seed=1,
        tolerance=0.05,
    )

    output = capsys.readouterr().out
    assert "Target alpha_LiF not reached" in output
    assert "Skipping alpha_LiLi tuning" in output
    assert "Both target alpha and alpha_LiLi reached" not in output
    assert status["alpha_reached"] is False
    assert status["alpha_LiLi_reached"] is True
    assert status["all_reached"] is False


def test_run_supports_legacy_tol_alias():
    sro = _fake_sro(alpha=-0.16, alpha_lili=0.03)

    status = sro.run(
        max_steps=0,
        target_alpha=-0.2,
        target_alpha_LiLi=0.0,
        random_seed=1,
        tol=0.05,
    )

    assert status["tolerance"] == 0.05
    assert status["all_reached"] is True
