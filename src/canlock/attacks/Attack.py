from abc import ABC, abstractmethod

class Attack(ABC):

    def init(self, signalName):
        self.signalName = signalName

    @abstractmethod
    def inject(self, dataFrame):
        pass

    @abstractmethod
    def getAttackName(self):
        pass
