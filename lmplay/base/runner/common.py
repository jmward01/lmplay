from abc import ABC, abstractmethod


class Component(ABC):
  """Wrapper for saveable/loadable sections of a runner.

  A Section encapsulates the operations for one saveable component:
  - archive(): Save current state
  - advertise(): Expose construction parameters
  - construct(): Build/rebuild the component

  """

  def __init__(self):
    pass

  @abstractmethod
  def archive(self):
    pass

  @abstractmethod
  def advertise(self):
    pass

  @abstractmethod
  def construct(self, construction_args: dict, state_args: dict, state: dict):
    pass

class NOPComponent(Component):

  def archive(self):
    pass

  def advertise(self):
    pass

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    pass


