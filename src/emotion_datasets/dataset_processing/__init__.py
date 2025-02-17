from .base import get_dataset

# These lines exist to ensure the class is registered
# Otherwise, the 'get_dataset' function can't find them
from .affective_text import AffectiveTextProcessor  # noqa: F401
from .carer import CARERProcessor  # noqa: F401
from .crowdflower import CrowdFlowerProcessor  # noqa: F401
from .emobank import EmoBankProcessor  # noqa: F401
from .emoint import EmoIntProcessor  # noqa: F401
from .fb_valence_arousal import FBValenceArousalProcessor  # noqa: F401
from .go_emotions import GoEmotionsProcessor  # noqa: F401
from .sentimental_liar import SentimentalLIARProcessor  # noqa: F401
from .ssec import SSECProcessor  # noqa: F401
from .tales_emotions import TalesEmotionsProcessor  # noqa: F401
from .xed import XEDProcessor  # noqa: F401

__all__ = ["get_dataset"]
