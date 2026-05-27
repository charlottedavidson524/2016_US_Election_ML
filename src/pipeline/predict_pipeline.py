import sys 
import os
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Map HTML inputs to backend 
    """
    def __init__(
        self,
        state_electoral_integrity: float,
        national_electoral_integrity: float,
        political_scale: float,
        election_rigged: int,
        deadlines_restrictive: int,
        voters_waited: int,
        voters_intimidated: int,
        multiple_ballots_cast: int,
        machines_accurate: int,
        records_secure: int,
        votes_counted_quickly: int,
        outcome_reflected_popular_will: int
    ):
        
        self.state_electoral_integrity = state_electoral_integrity
        self.national_electoral_integrity = national_electoral_integrity
        self.political_scale = political_scale
        self.election_rigged = election_rigged
        self.deadlines_restrictive = deadlines_restrictive
        self.voters_waited = voters_waited
        self.voters_intimidated = voters_intimidated
        self.multiple_ballots_cast = multiple_ballots_cast
        self.machines_accurate = machines_accurate
        self.records_secure = records_secure
        self.votes_counted_quickly = votes_counted_quickly
        self.outcome_reflected_popular_will = outcome_reflected_popular_will

    def get_data_as_data_frame(self):
        """
        Model was trained using dataframes so must format input data as df
        """
        try:
            custom_data_input_dict = {
                "State Electoral Integrity": [self.state_electoral_integrity],
                "National Electoral Integrity": [self.national_electoral_integrity],
                "Political Scale": [self.political_scale],
                "Election Rigged_encoded": [self.election_rigged],
                "Deadlines Restrictive_encoded": [self.deadlines_restrictive],
                "Voters Waited_encoded": [self.voters_waited],
                "Voters Intimidated_encoded": [self.voters_intimidated],
                "Multiple Ballots Cast_encoded": [self.multiple_ballots_cast],
                "Machines Accurate_encoded": [self.machines_accurate],
                "Records Secure_encoded": [self.records_secure],
                "Votes Counted Quickly_encoded": [self.votes_counted_quickly],
                "Outcome Reflected Popular Will_encoded": [self.outcome_reflected_popular_will]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

