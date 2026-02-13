from abc import ABC, abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def tear_down(self):
        pass
