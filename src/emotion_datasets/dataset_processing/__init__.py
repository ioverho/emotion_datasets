from .base import get_dataset

# These lines exist to ensure the class is registered
# Otherwise, the 'get_dataset' function can't find them
from .affective_text import AffectiveTextProcessor  # noqa: F401
from .canceremo import CancerEmoProcessor  # noqa: F401
from .carer import CARERProcessor  # noqa: F401
from .crowdflower import CrowdFlowerProcessor  # noqa: F401
from .electoral_tweets import ElectoralTweetsProcessor  # noqa: F401
from .emobank import EmoBankProcessor  # noqa: F401
from .emoint import EmoIntProcessor  # noqa: F401
from .emotion_stimulus import EmotionStimulusProcessor  # noqa: F401
from .fb_valence_arousal import FBValenceArousalProcessor  # noqa: F401
from .go_emotions import GoEmotionsProcessor  # noqa: F401
from .goodnewseveryone import GoodNewsEveryoneProcessor  # noqa: F401
from .hurricanes8 import Hurricanes8Processor  # noqa: F401
from .hurricanes24 import Hurricanes24Processor  # noqa: F401
from .isear import ISEARProcessor  # noqa: F401
from .ren20k import REN20kProcessor  # noqa: F401
from .semeval2018_classification import Semeval2018ClassificationProcessor  # noqa: F401
from .semeval2018_intensity import Semeval2018IntensityProcessor  # noqa: F401
from .sentimental_liar import SentimentalLIARProcessor  # noqa: F401
from .ssec import SSECProcessor  # noqa: F401
from .stockemotions import StockEmotionsProcessor  # noqa: F401
from .tales_emotions import TalesEmotionsProcessor  # noqa: F401
from .tec import TECProcessor  # noqa: F401
from .usvsthem import UsVsThemProcessor  # noqa: F401
from .wassa22 import WASSA22Processor  # noqa: F401
from .xed import XEDProcessor  # noqa: F401
from .debug import DebugProcessor  # noqa: F401

__all__ = ["get_dataset"]
