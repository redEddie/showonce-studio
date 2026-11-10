"""A continuation must not silently use a parent learner or normalizer/config."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest

path=Path(__file__).parents[1]/'scripts/export_show108.py'
spec=importlib.util.spec_from_file_location('export_show108_test',path)
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


@pytest.mark.parametrize('release',['16u','obsdr-wide','final24'])
def test_verified_learner_and_config_must_match_selected_release(release):
    expected=export.RELEASES[release]
    def digest(path):
        return expected['learner_sha256']if path.name=='learner.msgpack'else expected['config_sha256']
    with patch.object(export.subprocess,'check_output',return_value=expected['commit']),patch.object(export,'digest',side_effect=digest):
        _,_,checkpoint=export.verify_release(Path('/source'),Path('/run'),release)
    assert checkpoint.name==f"{expected['step']:012d}"
    assert expected['cumulative']==expected['parent']+expected['step']


@pytest.mark.parametrize('changed',['source','learner','configuration'])
def test_wrong_source_or_mixed_files_are_rejected(changed):
    expected=export.RELEASES['obsdr-wide']
    def digest(path):
        key='learner_sha256'if path.name=='learner.msgpack'else 'config_sha256'
        if (changed=='learner'and key=='learner_sha256')or(changed=='configuration'and key=='config_sha256'):
            return export.RELEASES['16u'][key]
        return expected[key]
    with patch.object(export.subprocess,'check_output',return_value='wrong'if changed=='source'else expected['commit']),patch.object(export,'digest',side_effect=digest):
        with pytest.raises(ValueError):export.verify_release(Path('/source'),Path('/run'),'obsdr-wide')
