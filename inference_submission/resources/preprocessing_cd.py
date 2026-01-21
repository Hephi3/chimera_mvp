import numpy as np

CD_KEYS = ["age", "sex", "smoking", "tumor", "stage", "substage", "grade", "reTUR", "LVI", "variant", "EORTC", "no_instillations", "BRS"]

def preproc_cd_file(cd: dict) -> dict:    
    # age: Normalize to 0-1, -> age / 100. Statistally age could be > 100 but is very rare
    if "age" not in cd or not isinstance(cd["age"], (int, float)) or cd["age"] < 0 or cd["age"] > 150:
        cd["age"] = 0  # Default to 0 if age is missing or invalid
    cd["age"] = np.clip(cd["age"], 0, 100) / 100
    
    # sex: 0: f, 1: m
    if "sex" not in cd or cd["sex"] not in ["Female", "Male"]:
        cd["sex"] = "Male"
    cd["sex"] = int(cd["sex"] == "Male")    
    
    # smoking: One-hot encoding: [Missing, No, Yes] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    if "smoking" in cd:
        if cd["smoking"] == "No":
            cd["smoking"] = [0, 1, 0]
        elif cd["smoking"] == "Yes":
            cd["smoking"] = [0, 0, 1]
        elif cd["smoking"] == "-1":
            cd["smoking"] = [1, 0, 0]
    else:
        cd["smoking"] = [1, 0, 0]
    
    # tumor: 0: Primary, 1: Recurrence
    if "tumor" not in cd or cd["tumor"] not in ["Primary", "Recurrence"]:
        cd["tumor"] = "Primary"
    cd["tumor"] = int(cd["tumor"] == "Recurrence")
    
    # stage: One-hot encoding: [T1HG, T2HG, T3HG, TaHG] -> [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    if "stage" not in cd or cd["stage"] not in ["T1HG", "T2HG", "T3HG", "TaHG"]:
        cd["stage"] = "T1HG"
    cd["stage"] = [int(cd["stage"] == "T1HG"), int(cd["stage"] == "T2HG"), int(cd["stage"] == "T3HG")]
    
    # substage: 0: T1e, 1: T1m
    # There are missing values -> Instead one-hot encoding: [Missing, T1e, T1m] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    assert "substage" not in cd or cd["substage"] in ["T1e", "T1m", "-1"], f"Substage {cd['substage']} is not valid"
    if "substage" in cd:
        if cd["substage"] == "T1e":
            cd["substage"] = [0, 1, 0]
        elif cd["substage"] == "T1m":
            cd["substage"] = [0, 0, 1]
        elif cd["substage"] == "-1":
            cd["substage"] = [1, 0, 0]
    else:
        cd["substage"] = [1, 0, 0]
    
    # grade: 0: G2, 1: G3
    if "grade" not in cd or cd["grade"] not in ["G2", "G3"]:
        cd["grade"] = "G2"
    cd["grade"] = int(cd["grade"] == "G3")
    
    # reTUR: 0: No, 1: Yes
    if "reTUR" not in cd or cd["reTUR"] not in ["No", "Yes"]:
        cd["reTUR"] = "No"
    cd["reTUR"] = int(cd["reTUR"] == "Yes")
    
    # LVI: 0: No, 1: Yes
    # There are missing values -> Instead one-hot encoding: [Missing, No, Yes] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    assert "LVI" not in cd or cd["LVI"] in ["No", "Yes", "-1"], f"LVI {cd['LVI']} is not valid"
    if "LVI" in cd:
        if cd["LVI"] == "No":
            cd["LVI"] = [0, 1, 0]
        elif cd["LVI"] == "Yes":
            cd["LVI"] = [0, 0, 1]
        elif cd["LVI"] == "-1":
            cd["LVI"] = [1, 0, 0]
    else:
        cd["LVI"] = [1, 0, 0]
    
    # variant: 0: UCC, 1: UCC + Variant
    if "variant" not in cd or cd["variant"] not in ["UCC", "UCC + Variant"]:
        cd["variant"] = "UCC"
    cd["variant"] = int(cd["variant"] == "UCC + Variant")
    
    # EORTC: 0: High risk, 1: Highest risk
    # There are missing values -> Instead one-hot encoding: [Missing, High risk, Highest risk] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    if "EORTC" in cd:
        if cd["EORTC"] == "High risk":
            cd["EORTC"] = [0, 1, 0]
        elif cd["EORTC"] == "Highest risk":
            cd["EORTC"] = [0, 0, 1]
        elif cd["EORTC"] == "-1":
            cd["EORTC"] = [1, 0, 0]
    else:
        cd["EORTC"] = [1, 0, 0]
    
    # no_instillations: Use tanh to normalize as it is not bound: np.tanh(0.05*np.maximum(0, cd["no_instillations"])), also indicate if it is missin (-1), they can also be just missing
    if "no_instillations" not in cd: cd["no_instillations"] = -1
    cd["no_instillations"] = [np.tanh(0.05*np.maximum(0, cd["no_instillations"])), int(cd["no_instillations"] == -1)] #np.log(cd["no_instillations"] + 1)
    
    if "BRS" in cd: del cd["BRS"]
    # Delete all other keys:
    for key in list(cd.keys()):
        if key not in CD_KEYS:
            del cd[key]
    # This will delete e.g. "chimera_id_t3" which is part of the validation/test set
    
    return cd
