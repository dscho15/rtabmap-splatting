from peewee import *
import datetime
from pathlib import Path

class RTABSQliteDatabase:
    
    
    def __init__(self, p_db: str):
        
        path = Path(p_db)
        
        # chekc if the database file exists
        if not path.exists():
            raise FileNotFoundError(f"Database file {p_db} not found")
        
        # Initialize the database
        self.db = SqliteDatabase(p_db)
        self.db.connect()
    
        # Define the BaseModel
        class BaseModel(Model):
            class Meta:
                database = self.db
                
        class Feature(BaseModel):
            node_id = IntegerField()
            word_id = IntegerField()
            pos_x = IntegerField()
            pos_y = IntegerField()
            size = IntegerField()
            dir = IntegerField()
            response = FloatField()
            octave = IntegerField()
            depth_x = FloatField()
            depth_y = FloatField()
            depth_z = FloatField()
            descriptor_size = IntegerField()
            descriptor = BlobField()
            

        # Define the Data model
        class Data(BaseModel):
            id = IntegerField(primary_key=True)
            image = BlobField()
            depth = BlobField()
            calibration = BlobField()
            scan = BlobField()
            scan_info = BlobField()
            ground_cells = BlobField()
            obstacle_cells = BlobField()
            empty_cells = BlobField()
            cell_size = FloatField()
            view_point_x = FloatField()
            view_point_y = FloatField()
            view_point_z = FloatField()
            user_data = BlobField()
            time_enter = DateTimeField(default=datetime.datetime.now)


        class Node(BaseModel):
            id = IntegerField(primary_key=True)
            map_id = IntegerField()
            weight = IntegerField()
            stamp = FloatField()
            pose = BlobField()
            ground_truth_pose = BlobField()
            velocity = BlobField()
            label = TextField()
            gps = BlobField()
            env_sensors = BlobField()
            time_enter = DateTimeField(default=datetime.datetime.now)
            
        self.models = {
            "Feature": Feature,
            "Data": Data,
            "Node": Node
        }
            
    @staticmethod
    def extract_rows_from_table(model):
        return list(model.select())
    
    # close model
    def close(self):
        self.db.close()
        
    def get_rows(self, model_name: str):
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        return self.extract_rows_from_table(model)