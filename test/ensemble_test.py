from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import re

# Raw pasted data
raw_text_test5 = """
interface_103 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_118 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_123 BRS3  1.0   1.0   1.0   0.6000000238418579
interface_128 BRS1  0.0   0.0   0.0   0.699999988079071
interface_129 BRS3  1.0   0.800000011920929 1.0   0.30000001192092896
interface_150 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_153 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_2  BRS2  0.0   0.0   0.0   0.4000000059604645
interface_22 BRS2  0.0   0.20000000298023224 0.10000000149011612 0.30000001192092896
interface_45 BRS3  0.9000000357627869 0.800000011920929 0.800000011920929 0.4000000059604645
interface_48 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_49 BRS2  0.9000000357627869 1.0   1.0   0.4000000059604645
interface_50 BRS1  1.0   0.699999988079071 0.800000011920929 0.0  
interface_54 BRS1  0.0   0.0   0.0   0.4000000059604645
interface_79 BRS2  0.4000000059604645 0.30000001192092896 0.5   0.10000000149011612
interface_90 BRS2  1.0   1.0   1.0   0.4000000059604645
"""

raw_text_test2 = """
interface_105 BRS1  0.0   0.0   0.0   0.30000001192092896
interface_12 BRS3  0.699999988079071 0.30000001192092896 0.20000000298023224 0.10000000149011612
interface_136 BRS2  0.5   0.20000000298023224 0.30000001192092896 0.0  
interface_145 BRS1  0.0   0.10000000149011612 0.0   0.699999988079071
interface_157 BRS1  0.0   0.0   0.0   0.0  
interface_158 BRS2  0.0   0.0   0.0   0.0  
interface_170 BRS3  1.0   1.0   1.0   0.6000000238418579
interface_19 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_29 BRS3  0.30000001192092896 0.4000000059604645 0.5   0.5  
interface_34 BRS3  1.0   0.800000011920929 1.0   0.0  
interface_38 BRS1  0.0   0.0   0.0   0.20000000298023224
interface_52 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_61 BRS2  0.4000000059604645 0.30000001192092896 0.30000001192092896 0.10000000149011612
interface_69 BRS2  0.0   0.0   0.0   0.5  
interface_92 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_95 BRS1  0.5   0.20000000298023224 0.4000000059604645 0.30000001192092896
"""

raw_text_test = """
interface_131 BRS1  0.9000000357627869 0.5   0.800000011920929 0.0  
interface_139 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_154 BRS1  1.0   1.0   1.0   0.10000000149011612
interface_30 BRS2  0.0   0.0   0.0   0.0  
interface_37 BRS2  0.0   0.0   0.0   0.0  
interface_38 BRS1  0.0   0.0   0.0   0.20000000298023224
interface_40 BRS3  1.0   1.0   0.9000000357627869 0.0  
interface_45 BRS3  0.9000000357627869 0.800000011920929 0.800000011920929 0.4000000059604645
interface_50 BRS1  1.0   0.699999988079071 0.800000011920929 0.0  
interface_51 BRS3  0.9000000357627869 0.800000011920929 0.9000000357627869 0.30000001192092896
interface_61 BRS2  0.4000000059604645 0.30000001192092896 0.30000001192092896 0.10000000149011612
interface_68 BRS3  0.0   0.0   0.0   0.30000001192092896
interface_78 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_90 BRS2  1.0   1.0   1.0   0.4000000059604645
interface_92 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_94 BRS3  1.0   1.0   1.0   0.10000000149011612
"""


raw_text_all = """
interface_0  BRS2  0.0   0.0   0.0   0.10000000149011612
interface_1  BRS3  1.0   1.0   1.0   0.30000001192092896
interface_10 BRS2  0.9000000357627869 0.6000000238418579 0.800000011920929 0.20000000298023224
interface_100 BRS2  0.0   0.0   0.0   0.0  
interface_101 BRS1  0.800000011920929 0.6000000238418579 0.699999988079071 0.0  
interface_102 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_103 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_104 BRS1  1.0   1.0   1.0   0.20000000298023224
interface_105 BRS1  0.0   0.0   0.0   0.30000001192092896
interface_106 BRS3  0.800000011920929 0.6000000238418579 0.800000011920929 0.30000001192092896
interface_107 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_108 BRS1  0.0   0.0   0.0   0.0  
interface_109 BRS1  0.0   0.0   0.10000000149011612 0.5  
interface_11 BRS1  0.0   0.0   0.0   0.4000000059604645
interface_110 BRS1  0.10000000149011612 0.0   0.10000000149011612 0.0  
interface_111 BRS1  0.0   0.0   0.0   0.30000001192092896
interface_112 BRS1  0.10000000149011612 0.20000000298023224 0.20000000298023224 0.0  
interface_113 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_114 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_115 BRS3  1.0   0.9000000357627869 0.9000000357627869 0.4000000059604645
interface_116 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_117 BRS1  0.9000000357627869 0.9000000357627869 0.9000000357627869 0.20000000298023224
interface_118 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_119 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_12 BRS3  0.699999988079071 0.30000001192092896 0.20000000298023224 0.10000000149011612
interface_120 BRS1  0.0   0.10000000149011612 0.20000000298023224 0.30000001192092896
interface_121 BRS1  0.4000000059604645 0.5   0.5   0.10000000149011612
interface_122 BRS1  0.0   0.0   0.0   0.0  
interface_123 BRS3  1.0   1.0   1.0   0.6000000238418579
interface_124 BRS1  1.0   0.6000000238418579 0.9000000357627869 0.0  
interface_125 BRS3  0.0   0.0   0.0   0.20000000298023224
interface_126 BRS2  0.0   0.0   0.0   0.20000000298023224
interface_127 BRS2  0.5   0.10000000149011612 0.30000001192092896 0.0  
interface_128 BRS1  0.0   0.0   0.0   0.699999988079071
interface_129 BRS3  1.0   0.800000011920929 1.0   0.30000001192092896
interface_13 BRS3  0.699999988079071 0.5   0.699999988079071 0.0  
interface_130 BRS1  0.10000000149011612 0.0   0.0   0.0  
interface_131 BRS1  0.9000000357627869 0.5   0.800000011920929 0.0  
interface_132 BRS2  0.0   0.0   0.0   0.0  
interface_133 BRS3  1.0   1.0   1.0   0.5  
interface_134 BRS2  0.0   0.0   0.0   0.0  
interface_135 BRS1  0.5   0.10000000149011612 0.10000000149011612 0.0  
interface_136 BRS2  0.5   0.20000000298023224 0.30000001192092896 0.0  
interface_137 BRS2  0.0   0.0   0.0   0.0  
interface_138 BRS3  1.0   0.699999988079071 0.800000011920929 0.20000000298023224
interface_139 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_14 BRS3  0.30000001192092896 0.30000001192092896 0.30000001192092896 0.20000000298023224
interface_140 BRS2  0.0   0.0   0.0   0.0  
interface_141 BRS2  0.0   0.0   0.0   0.0  
interface_142 BRS2  0.0   0.0   0.0   0.0  
interface_143 BRS2  1.0   1.0   1.0   0.0  
interface_144 BRS1  0.0   0.0   0.0   0.30000001192092896
interface_145 BRS1  0.0   0.10000000149011612 0.0   0.699999988079071
interface_146 BRS2  0.0   0.0   0.0   0.0  
interface_147 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_148 BRS3  0.30000001192092896 0.0   0.20000000298023224 0.10000000149011612
interface_149 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_15 BRS1  0.0   0.0   0.0   0.5  
interface_150 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_151 BRS2  0.4000000059604645 0.10000000149011612 0.20000000298023224 0.0  
interface_152 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_153 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_154 BRS1  1.0   1.0   1.0   0.10000000149011612
interface_155 BRS1  0.0   0.0   0.0   0.20000000298023224
interface_156 BRS2  0.0   0.0   0.0   0.0  
interface_157 BRS1  0.0   0.0   0.0   0.0  
interface_158 BRS2  0.0   0.0   0.0   0.0  
interface_159 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_16 BRS1  0.0   0.10000000149011612 0.10000000149011612 0.30000001192092896
interface_160 BRS1  0.0   0.0   0.0   0.20000000298023224
interface_161 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_162 BRS2  0.0   0.10000000149011612 0.0   0.10000000149011612
interface_163 BRS2  0.0   0.0   0.0   0.0  
interface_164 BRS3  0.6000000238418579 0.4000000059604645 0.5   0.10000000149011612
interface_165 BRS2  0.30000001192092896 0.0   0.10000000149011612 0.30000001192092896
interface_166 BRS3  1.0   1.0   1.0   0.20000000298023224
interface_167 BRS1  1.0   0.9000000357627869 0.9000000357627869 0.4000000059604645
interface_168 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_169 BRS1  0.0   0.0   0.0   0.4000000059604645
interface_17 BRS2  0.800000011920929 0.699999988079071 0.9000000357627869 0.0  
interface_170 BRS3  1.0   1.0   1.0   0.6000000238418579
interface_171 BRS1  0.0   0.0   0.0   0.0  
interface_172 BRS3  1.0   1.0   1.0   0.800000011920929
interface_173 BRS1  0.0   0.0   0.0   0.4000000059604645
interface_174 BRS1  0.0   0.0   0.0   0.5  
interface_175 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_176 BRS3  0.800000011920929 0.800000011920929 0.800000011920929 0.0  
interface_177 BRS3  1.0   1.0   1.0   0.5  
interface_178 BRS3  0.800000011920929 0.6000000238418579 0.800000011920929 0.10000000149011612
interface_179 BRS2  0.0   0.0   0.0   0.0  
interface_18 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_180 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_19 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_2  BRS2  0.0   0.0   0.0   0.4000000059604645
interface_20 BRS2  0.0   0.0   0.0   0.30000001192092896
interface_21 BRS2  0.20000000298023224 0.0   0.20000000298023224 0.10000000149011612
interface_22 BRS2  0.0   0.20000000298023224 0.10000000149011612 0.30000001192092896
interface_23 BRS2  0.0   0.0   0.0   0.30000001192092896
interface_24 BRS2  0     0     0     0    
interface_25 BRS3  1.0   0.800000011920929 1.0   0.800000011920929
interface_26 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_27 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_28 BRS3  1.0   0.9000000357627869 1.0   0.6000000238418579
interface_29 BRS3  0.30000001192092896 0.4000000059604645 0.5   0.5  
interface_3  BRS2  0.0   0.0   0.0   0.4000000059604645
interface_30 BRS2  0.0   0.0   0.0   0.0  
interface_31 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_32 BRS1  0.0   0.0   0.0   0.4000000059604645
interface_33 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_34 BRS3  1.0   0.800000011920929 1.0   0.0  
interface_35 BRS2  0.0   0.0   0.0   0.20000000298023224
interface_36 BRS2  0.0   0.0   0.0   0.6000000238418579
interface_37 BRS2  0.0   0.0   0.0   0.0  
interface_38 BRS1  0.0   0.0   0.0   0.20000000298023224
interface_39 BRS2  0.0   0.0   0.0   0.20000000298023224
interface_4  BRS1  0.20000000298023224 0.10000000149011612 0.30000001192092896 0.0  
interface_40 BRS3  1.0   1.0   0.9000000357627869 0.0  
interface_41 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_42 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_43 BRS2  1.0   1.0   1.0   0.10000000149011612
interface_44 BRS3  1.0   0.699999988079071 0.800000011920929 0.10000000149011612
interface_45 BRS3  0.9000000357627869 0.800000011920929 0.800000011920929 0.4000000059604645
interface_46 BRS3  1.0   1.0   1.0   0.20000000298023224
interface_47 BRS3  1.0   1.0   1.0   0.6000000238418579
interface_48 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_49 BRS2  0.9000000357627869 1.0   1.0   0.4000000059604645
interface_5  BRS1  0.0   0.0   0.0   0.20000000298023224
interface_50 BRS1  1.0   0.699999988079071 0.800000011920929 0.0  
interface_51 BRS3  0.9000000357627869 0.800000011920929 0.9000000357627869 0.30000001192092896
interface_52 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_53 BRS2  0.0   0.0   0.0   0.0  
interface_54 BRS1  0.0   0.0   0.0   0.4000000059604645
interface_55 BRS3  1.0   0.9000000357627869 0.9000000357627869 0.5  
interface_56 BRS2  0.0   0.0   0.0   0.30000001192092896
interface_57 BRS3  0.0   0.0   0.0   0.20000000298023224
interface_58 BRS2  0.9000000357627869 1.0   1.0   0.4000000059604645
interface_59 BRS3  0.20000000298023224 0.20000000298023224 0.30000001192092896 0.4000000059604645
interface_6  BRS2  0.10000000149011612 0.10000000149011612 0.10000000149011612 0.10000000149011612
interface_60 BRS3  1.0   0.800000011920929 1.0   0.6000000238418579
interface_61 BRS2  0.4000000059604645 0.30000001192092896 0.30000001192092896 0.10000000149011612
interface_62 BRS2  0.9000000357627869 0.800000011920929 0.9000000357627869 0.30000001192092896
interface_63 BRS1  0.0   0.0   0.0   0.5  
interface_64 BRS2  0.6000000238418579 0.20000000298023224 0.20000000298023224 0.20000000298023224
interface_65 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_66 BRS2  0.10000000149011612 0.0   0.0   0.10000000149011612
interface_67 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_68 BRS3  0.0   0.0   0.0   0.30000001192092896
interface_69 BRS2  0.0   0.0   0.0   0.5  
interface_7  BRS2  0.5   0.5   0.699999988079071 0.0  
interface_70 BRS3  1.0   1.0   1.0   0.0  
interface_71 BRS1  0     0     0     0    
interface_72 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_73 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_74 BRS2  0     0     0     0    
interface_75 BRS2  0.0   0.0   0.0   0.30000001192092896
interface_76 BRS2  0.0   0.0   0.0   0.0  
interface_77 BRS3  0.9000000357627869 0.800000011920929 0.9000000357627869 0.20000000298023224
interface_78 BRS1  0.0   0.0   0.0   0.10000000149011612
interface_79 BRS2  0.4000000059604645 0.30000001192092896 0.5   0.10000000149011612
interface_8  BRS1  0.5   0.4000000059604645 0.4000000059604645 0.20000000298023224
interface_80 BRS2  0.9000000357627869 0.800000011920929 0.800000011920929 0.10000000149011612
interface_81 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_82 BRS3  0.800000011920929 0.5   0.5   0.6000000238418579
interface_83 BRS3  1.0   1.0   0.9000000357627869 0.30000001192092896
interface_84 BRS2  0.0   0.0   0.0   0.0  
interface_85 BRS2  0.0   0.0   0.0   0.10000000149011612
interface_86 BRS3  1.0   1.0   1.0   0.800000011920929
interface_87 BRS1  0.0   0.0   0.0   0.0  
interface_88 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_89 BRS3  1.0   0.800000011920929 0.9000000357627869 0.10000000149011612
interface_9  BRS2  0.0   0.0   0.0   0.10000000149011612
interface_90 BRS2  1.0   1.0   1.0   0.4000000059604645
interface_91 BRS2  0.0   0.0   0.0   0.0  
interface_92 BRS2  0.0   0.0   0.0   0.4000000059604645
interface_93 BRS1  0.0   0.0   0.0   0.0  
interface_94 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_95 BRS1  0.5   0.20000000298023224 0.4000000059604645 0.30000001192092896
interface_96 BRS3  1.0   1.0   1.0   0.5  
interface_97 BRS1  0.699999988079071 0.30000001192092896 0.5   0.6000000238418579
interface_98 BRS1  0.6000000238418579 0.6000000238418579 0.699999988079071 0.10000000149011612
interface_99 BRS1  0.20000000298023224 0.20000000298023224 0.20000000298023224 0.10000000149011612
"""
raw_text_differences = """
interface_1  BRS3  1.0   1.0   1.0   0.30000001192092896
interface_10 BRS2  0.9000000357627869 0.6000000238418579 0.800000011920929 0.20000000298023224
interface_101 BRS1  0.800000011920929 0.6000000238418579 0.699999988079071 0.0  
interface_102 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_104 BRS1  1.0   1.0   1.0   0.20000000298023224
interface_106 BRS3  0.800000011920929 0.6000000238418579 0.800000011920929 0.30000001192092896
interface_107 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_113 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_115 BRS3  1.0   0.9000000357627869 0.9000000357627869 0.4000000059604645
interface_117 BRS1  0.9000000357627869 0.9000000357627869 0.9000000357627869 0.20000000298023224
interface_118 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_12 BRS3  0.699999988079071 0.30000001192092896 0.20000000298023224 0.10000000149011612
interface_124 BRS1  1.0   0.6000000238418579 0.9000000357627869 0.0  
interface_125 BRS3  0.0   0.0   0.0   0.20000000298023224
interface_128 BRS1  0.0   0.0   0.0   0.699999988079071
interface_129 BRS3  1.0   0.800000011920929 1.0   0.30000001192092896
interface_13 BRS3  0.699999988079071 0.5   0.699999988079071 0.0  
interface_131 BRS1  0.9000000357627869 0.5   0.800000011920929 0.0  
interface_133 BRS3  1.0   1.0   1.0   0.5  
interface_138 BRS3  1.0   0.699999988079071 0.800000011920929 0.20000000298023224
interface_14 BRS3  0.30000001192092896 0.30000001192092896 0.30000001192092896 0.20000000298023224
interface_143 BRS2  1.0   1.0   1.0   0.0  
interface_145 BRS1  0.0   0.10000000149011612 0.0   0.699999988079071
interface_147 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_148 BRS3  0.30000001192092896 0.0   0.20000000298023224 0.10000000149011612
interface_154 BRS1  1.0   1.0   1.0   0.10000000149011612
interface_164 BRS3  0.6000000238418579 0.4000000059604645 0.5   0.10000000149011612
interface_166 BRS3  1.0   1.0   1.0   0.20000000298023224
interface_167 BRS1  1.0   0.9000000357627869 0.9000000357627869 0.4000000059604645
interface_17 BRS2  0.800000011920929 0.699999988079071 0.9000000357627869 0.0  
interface_175 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_176 BRS3  0.800000011920929 0.800000011920929 0.800000011920929 0.0  
interface_177 BRS3  1.0   1.0   1.0   0.5  
interface_178 BRS3  0.800000011920929 0.6000000238418579 0.800000011920929 0.10000000149011612
interface_18 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_180 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_29 BRS3  0.30000001192092896 0.4000000059604645 0.5   0.5  
interface_31 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_34 BRS3  1.0   0.800000011920929 1.0   0.0  
interface_36 BRS2  0.0   0.0   0.0   0.6000000238418579
interface_40 BRS3  1.0   1.0   0.9000000357627869 0.0  
interface_42 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_43 BRS2  1.0   1.0   1.0   0.10000000149011612
interface_44 BRS3  1.0   0.699999988079071 0.800000011920929 0.10000000149011612
interface_45 BRS3  0.9000000357627869 0.800000011920929 0.800000011920929 0.4000000059604645
interface_46 BRS3  1.0   1.0   1.0   0.20000000298023224
interface_48 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_49 BRS2  0.9000000357627869 1.0   1.0   0.4000000059604645
interface_50 BRS1  1.0   0.699999988079071 0.800000011920929 0.0  
interface_51 BRS3  0.9000000357627869 0.800000011920929 0.9000000357627869 0.30000001192092896
interface_52 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_55 BRS3  1.0   0.9000000357627869 0.9000000357627869 0.5  
interface_57 BRS3  0.0   0.0   0.0   0.20000000298023224
interface_58 BRS2  0.9000000357627869 1.0   1.0   0.4000000059604645
interface_59 BRS3  0.20000000298023224 0.20000000298023224 0.30000001192092896 0.4000000059604645
interface_62 BRS2  0.9000000357627869 0.800000011920929 0.9000000357627869 0.30000000298023224
interface_64 BRS2  0.6000000238418579 0.20000000298023224 0.20000000298023224 0.20000000298023224
interface_68 BRS3  0.0   0.0   0.0   0.30000001192092896
interface_7  BRS2  0.5   0.5   0.699999988079071 0.0  
interface_70 BRS3  1.0   1.0   1.0   0.0  
interface_72 BRS3  1.0   1.0   1.0   0.30000001192092896
interface_73 BRS3  1.0   1.0   1.0   0.4000000059604645
interface_77 BRS3  0.9000000357627869 0.800000011920929 0.9000000357627869 0.20000000298023224
interface_80 BRS2  0.9000000357627869 0.800000011920929 0.800000011920929 0.10000000149011612
interface_82 BRS3  0.800000011920929 0.5   0.5   0.6000000238418579
interface_83 BRS3  1.0   1.0   0.9000000357627869 0.30000001192092896
interface_89 BRS3  1.0   0.800000011920929 0.9000000357627869 0.10000000149011612
interface_90 BRS2  1.0   1.0   1.0   0.4000000059604645
interface_94 BRS3  1.0   1.0   1.0   0.10000000149011612
interface_96 BRS3  1.0   1.0   1.0   0.5  
interface_97 BRS1  0.699999988079071 0.30000001192092896 0.5   0.6000000238418579
interface_98 BRS1  0.6000000238418579 0.6000000238418579 0.699999988079071 0.10000000149011612
"""

raw_text = raw_text_all

# Parse each line into parts: id, label, 4 probs
rows = []
for line in raw_text.strip().split("\n"):
    parts = line.split()
    if len(parts) >= 6:
        idx = parts[0]
        label = parts[1]
        probs = [float(x) for x in parts[2:]]
        # ensure exactly 4 columns
        while len(probs) < 4:
            probs.append(0.0)
        rows.append([idx, label] + probs[:4])

df = pd.DataFrame(rows, columns=["id", "label", "p_dunkelblau", "p_grün", "p_rot", "p_hellblau"])

# Binary label: BRS3=1, else 0
df["y_true"] = (df["label"] == "BRS3").astype(int)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

results = {}

# model_cols = ["p_dunkelblau", "p_grün", "p_rot", "p_hellblau"]
# model_cols = ["p_dunkelblau", "p_rot", "p_hellblau"]
model_cols = ["p_dunkelblau", "p_grün", "p_rot"]
for model in model_cols:
    probs = df[model]
    preds = (probs >= 0.5).astype(int)
    results[model] = {
        "accuracy": accuracy_score(df.y_true, preds),
        "precision": precision_score(df.y_true, preds, zero_division=0),
        "recall": recall_score(df.y_true, preds),
        "f1": f1_score(df.y_true, preds),
        "roc_auc": roc_auc_score(df.y_true, probs)
    }

    print(f"RESULTS FOR MODEL: {model}")
    print(f"Accuracy: {results[model]['accuracy']:.4f}")
    print(f"Precision: {results[model]['precision']:.4f}")
    print(f"Recall: {results[model]['recall']:.4f}")
    print(f"F1 Score: {results[model]['f1']:.4f}")
    print(f"ROC AUC: {results[model]['roc_auc']:.4f}")

# Rename first columns
# Assumes first two columns are interface and true_class, rest are fold probabilities
num_folds = df.shape[1] - 2
col_names = ['interface', 'true_class', 'p_dunkelblau', 'p_grün', 'p_rot', 'p_hellblau', 'y_true']
# df.columns = col_names

# Compute mean probability across all p_...
df['mean_prob'] = df[model_cols].mean(axis=1)

# Convert mean probability to binary prediction for BRS3
df['db_bin'] = (df['p_dunkelblau'] >= 0.5).astype(int)
df['->db?'] = np.where(df['y_true'] == df['db_bin'], "\u2713", "\u2717")

df['->bin'] = (df['mean_prob'] >= 0.5).astype(int)

# Add column to show if pred_BRS3 is correct by comparing y_true and pred_BRS3
df['->?'] = np.where(df['y_true'] == df['->bin'], "\u2713", "\u2717")

df['Improvement?'] = np.where((df['->db?'] != df['->?']), df['->?'], "-")

with pd.option_context('display.max_rows', None):  # more options can be specified also
    print(df)#[['interface', 'true_class', 'mean_prob', 'pred_BRS3']])


f1_columns = ['db_bin', '->bin']  # Add any other columns you want

for col in f1_columns:
    if col in df.columns:
        # Binarize if not already binary
        if df[col].nunique() > 2 or not set(df[col].unique()).issubset({0, 1}):
            binarized = (df[col] >= 0.5).astype(int)
        else:
            binarized = df[col]
        score = f1_score(df['y_true'], binarized) if binarized.nunique() > 1 else None
        print(f"F1 score for column '{col}' vs y_true: {score:.4f}" if score is not None else f"Column '{col}' has only one unique value, F1 not defined.")



from sklearn.metrics import f1_score
import numpy as np

best_f1 = 0
best_thresh = 0
for t in np.linspace(0, 1, 101):
    preds = (df['mean_prob'] >= t).astype(int)
    f1 = f1_score(df['y_true'], preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best threshold: {best_thresh}, F1: {best_f1:.4f}")

# weights = np.array([0.5, 0.2, 0.2, 0.1])  # e.g., more weight to p_dunkelblau
# df['weighted_prob'] = (df[model_cols] * weights).sum(axis=1)
# df['weighted_pred'] = (df['weighted_prob'] >= 0.5).astype(int)
# f1 = f1_score(df['y_true'], df['weighted_pred'])
# print(f"Weighted ensemble F1: {f1:.4f}")







# df_binary = (df[p_cols] >= 0.5).astype(int)
# df['majority_vote'] = (df_binary.sum(axis=1) >= 3).astype(int)  # at least 3/4 models vote BRS3
# f1 = f1_score(df['y_true'], df['majority_vote'])
# print(f"Majority vote F1: {f1:.4f}")

# def optimal_threshold(y_true, probs, thresholds=np.linspace(0, 1, 101)):
#     best_f1 = 0
#     best_thresh = 0.5
#     for t in thresholds:
#         preds = (probs >= t).astype(int)
#         f1 = f1_score(y_true, preds)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = t
#     return best_thresh, best_f1

# # Step 1: Find optimal thresholds
# optimal_threshs = {}
# for col in model_cols:
#     thresh, f1 = optimal_threshold(df['y_true'], df[col])
#     optimal_threshs[col] = thresh
#     print(f"Model {col}: optimal threshold={thresh:.2f}")

# # Step 2: Binarize predictions using optimal thresholds
# for col in model_cols:
#     df[col + '_bin'] = (df[col] >= optimal_threshs[col]).astype(int)

# # Step 3: Majority vote
# bin_cols = [c + '_bin' for c in model_cols]
# df['majority_vote'] = (df[bin_cols].sum(axis=1) > len(bin_cols)/2).astype(int)

# # Step 4: Compute F1 for majority vote
# f1_majority = f1_score(df['y_true'], df['majority_vote'])
# print(f"F1 score for majority vote with optimal thresholds: {f1_majority:.4f}")








# import numpy as np
# # --- Weighted ensemble ---
# # Define weights for each fold (must sum to 1)
# # Example: equal weights
# weights = np.ones(num_folds) / num_folds

# # Or custom weights, e.g., more weight to first fold: weights = np.array([0.4, 0.3, 0.2, 0.1])
# # Make sure length of weights == num_folds
# if len(weights) != num_folds:
#     raise ValueError("Length of weights must equal number of fold columns")

# # Compute weighted mean probability
# df['weighted_mean_prob'] = df[fold_cols].dot(weights)

# # Convert weighted mean probability to binary prediction for BRS3
# df['pred_BRS3'] = (df['weighted_mean_prob'] >= 0.5).astype(int)

# # Save results
# df[['interface', 'true_class', 'weighted_mean_prob', 'pred_BRS3']].to_csv(
#     'brs3_weighted_ensemble.csv', index=False
# )

# print(df[['interface', 'true_class', 'weighted_mean_prob', 'pred_BRS3']])



# # Assuming columns: interface, true_class, prob_model1, prob_model2, prob_model3, prob_model4
# df.columns = ['interface', 'true_class', 'model1', 'model2', 'model3', 'model4']

# model_cols = ['model1', 'model2', 'model3', 'model4']

# # Compute mean probability across models
# df['mean_prob'] = df[model_cols].mean(axis=1)

# # Convert true class to binary: 1 if BRS3, else 0
# df['true_binary'] = (df['true_class'] == 'BRS3').astype(int)

# # Ensemble prediction using mean probability
# df['ensemble_pred'] = (df['mean_prob'] >= 0.5).astype(int)

# # Majority vote prediction from individual model probabilities
# df_binary = (df[model_cols] >= 0.5).astype(int)
# df['majority_vote'] = df_binary.sum(axis=1).apply(lambda x: 1 if x > len(model_cols)/2 else 0)

# # Check if ensemble prediction changed compared to majority vote
# df['ensemble_changed'] = df['ensemble_pred'] != df['majority_vote']

# # Compute F1 score
# f1 = f1_score(df['true_binary'], df['ensemble_pred'])
# print(f"Weighted ensemble F1 score: {f1:.4f}")

# # Save results
# df[['interface', 'true_class', 'mean_prob', 'ensemble_pred', 'majority_vote', 'ensemble_changed']].to_csv(
#     'brs3_ensemble_vs_majority.csv', index=False
# )

# print(f"Weighted ensemble F1 score: {f1:.4f}")
# with pd.option_context('display.max_rows', None):  # more options can be specified also
#     print(df)

# # # Save results
# # df[['interface', 'true_class', 'mean_prob', 'ensemble_pred', 'majority_vote', 'ensemble_changed']].to_csv(
# #     'brs3_ensemble_vs_majority.csv', index=False
# # )

# Model p_dunkelblau: optimal threshold=0.61
# Model p_grün: optimal threshold=0.21
# Model p_rot: optimal threshold=0.41
# Model p_hellblau: optimal threshold=0.01
# F1 score for majority vote with optimal thresholds: 0.7874