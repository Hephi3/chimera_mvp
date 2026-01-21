import numpy as np

CD_KEYS = ["age", "sex", "smoking", "tumor", "stage", "substage", "grade", "reTUR", "LVI", "variant", "EORTC", "no_instillations", "BRS"]

def preproc_cd_file(cd: dict) -> dict:
    
    # age: Normalize to 0-1, -> age / 100. Statistally age could be > 100 but is very rare
    assert (isinstance(cd["age"], float)) and cd["age"] > 0 and cd["age"] < 150, f"Age {cd['age']} is not valid, {type(cd['age'])}"
    cd["age"] = np.clip(cd["age"], 0, 100) / 100
    
    # sex: 0: f, 1: m
    assert cd["sex"] in ["Female", "Male"], f"Sex {cd['sex']} is not valid"
    cd["sex"] = int(cd["sex"] == "Male")    
    
    # smoking: One-hot encoding: [Missing, No, Yes] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # assert cd["smoking"] in ["Missing", "No", "Yes"], f"Smoking {cd['smoking']} is not valid"
    # cd["smoking"] = [int(cd["smoking"] == "Missing"), int(cd["smoking"] == "No"), int(cd["smoking"] == "Yes")]
    # There are missing values -> Instead one-hot encoding: [Missing, No, Yes] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    assert "smoking" not in cd or cd["smoking"] in ["No", "Yes", "-1"], f"smoking {cd['smoking']} is not valid" 
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
    assert cd["tumor"] in ["Primary", "Recurrence"], f"Tumor {cd['tumor']} is not valid"
    cd["tumor"] = int(cd["tumor"] == "Recurrence")
    
    # stage: One-hot encoding: [T1HG, T2HG, T3HG, TaHG] -> [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    assert cd["stage"] in ["T1HG", "T2HG", "T3HG", "TaHG"], f"Stage {cd['stage']} is not valid"
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
    assert cd["grade"] in ["G2", "G3"], f"Grade {cd['grade']} is not valid"
    cd["grade"] = int(cd["grade"] == "G3")
    
    # reTUR: 0: No, 1: Yes
    assert cd["reTUR"] in ["No", "Yes"], f"reTUR {cd['reTUR']} is not valid"
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
    assert cd["variant"] in ["UCC", "UCC + Variant"], f"Variant {cd['variant']} is not valid"
    cd["variant"] = int(cd["variant"] == "UCC + Variant")
    
    # EORTC: 0: High risk, 1: Highest risk
    # There are missing values -> Instead one-hot encoding: [Missing, High risk, Highest risk] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    assert "EORTC" not in cd or cd["EORTC"] in ["High risk", "Highest risk", "-1"], f"EORTC {cd['EORTC']} is not valid"
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
    assert "no_instillations" not in cd or cd["no_instillations"] > 0 or cd["no_instillations"] == -1, f"no_instillations {cd['no_instillations']} is not valid"
    if "no_instillations" not in cd: cd["no_instillations"] = -1
    cd["no_instillations"] = [np.tanh(0.05*np.maximum(0, cd["no_instillations"])), int(cd["no_instillations"] == -1)] #np.log(cd["no_instillations"] + 1)
    
    # BRS: One-hot encoding: [BRS1, BRS2, BRS3] -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
    #    OR binary: 0: BRS3, 1; BRS1/2 as this is the criterion for the study: int(cd["BRS"] in ["BRS1", "BRS2"])
    assert cd["BRS"] in ["BRS1", "BRS2", "BRS3"], f"BRS {cd['BRS']} is not valid"
    cd["BRS"] = [int(cd["BRS"] == "BRS1"), int(cd["BRS"] == "BRS2"), int(cd["BRS"] == "BRS3")] # TODO: Predict binary in the end!!!
    
    # Delete all other keys:
    for key in list(cd.keys()):
        if key not in CD_KEYS:
            del cd[key]
    # This will delete e.g. "chimera_id_t3" which is part of the validation/test set
    
    return cd