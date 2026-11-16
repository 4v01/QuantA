from alphacontextlab.v10_clock.catalogue import original_title


def test_original_report_filter_rejects_summary_revision_and_wrong_year():
    assert original_title('平安银行：2019年年度报告',2019)
    assert original_title('2020 年年度报告（全文）',2020)
    assert not original_title('2019年年度报告摘要',2019)
    assert not original_title('2019年年度报告（修订版）',2019)
    assert not original_title('2020年年度报告',2019)
    assert not original_title('关于2019年年度报告的问询回复',2019)
