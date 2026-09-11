from enum import Enum
class ARCType(str,Enum):
 GRID='Grid'; OBJECT_SET='ObjectSet'; OBJECT='ARCObject'; OBJECT_PAIR='ObjectPair'; MASK='Mask'; REGION='Region'; REGION_SET='RegionSet'; PATH='Path'; PIXEL_GRAPH='PixelGraph'; SEQUENCE='Sequence'; RUN_LENGTH_SEQUENCE='RunLengthSequence'; FREQUENCY_MAP='FrequencyMap'; GRID_PAIR='GridPair'; BOOLEAN='Boolean'; INTEGER='Integer'; COLOR='Color'; COORDINATE='Coordinate'; GRID_SHAPE='GridShape'
