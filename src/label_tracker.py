import abc
import json


class LabelTracker(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_language_index') and callable(subclass.get_language_index) and
                hasattr(subclass, 'get_country_index') and callable(subclass.get_country_index) or
                NotImplemented)

    @abc.abstractmethod
    def get_language_index(self, language: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_country_index(self, country_code: str) -> int:
        raise NotImplementedError


class FileLabelTracker(LabelTracker):

    def __init__(self, languages_filename, country_codes_filename):
        with open(languages_filename, 'r') as f:
            self.languages = json.load(f)
            self.oov_lang_index = len(self.languages.keys())
            self.language_by_index = {v: k for k, v in self.languages.items()}
        with open(country_codes_filename, 'r') as f:
            self.geo_country_codes = json.load(f)
            self.oov_country_code_index = len(self.geo_country_codes.keys())
            self.country_by_index = {v: k for k, v in self.geo_country_codes.items()}

    def get_language_index(self, language):
        return self.languages[language] if language in self.languages.keys() else self.oov_lang_index

    def get_language(self, index):
        return self.language_by_index[index]

    def get_country_index(self, country_code):
        return self.geo_country_codes[country_code] if country_code in self.geo_country_codes.keys() \
            else self.oov_country_code_index

    def get_country(self, index):
        return self.country_by_index[index]


class DictLabelTracker(LabelTracker):
    """A container for labels with lazy registration"""

    def __init__(self):
        self.language_index = 0
        self.country_code_index = 0
        self.languages = {}
        self.geo_country_codes = {}

    def get_language_index(self, language):
        if language not in self.languages.keys():
            self.languages[language] = self.language_index
            self.language_index += 1
        return self.languages[language]

    def get_country_index(self, country_code):
        if country_code not in self.geo_country_codes.keys():
            self.geo_country_codes[country_code] = self.country_code_index
            self.country_code_index += 1
        return self.geo_country_codes[country_code]
