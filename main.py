import os
import logging
import random
import csv
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages connections and birthday logic"""
    
    def __init__(self, csv_file: str = "SFMBA Directory.csv"):
        self.csv_file = csv_file
        self.people = []
        self.user_connections = {}  # {name: connection_level}
        self.weights = {"Close": 3, "Normal": 1, "Stop": 0}
        self.load_data()
    
    def load_data(self):
        """Load people data from CSV"""
        try:
            with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # Map CSV fields to our expected format
                field_map = {
                    'Name': 'Name',
                    'Personal Email': 'Personal Email',
                    'MIT Email': 'MIT Email', 
                    'Birthday Month': 'Birthday Month',
                    'Birthday Day': 'Birthday Day'
                }
                
                self.people = []
                for row in reader:
                    try:
                        person = {}
                        for our_field, csv_field in field_map.items():
                            person[our_field] = row.get(csv_field, '').strip()
                        
                        if person['Name']:  # Only add if name exists
                            self.people.append(person)
                    except Exception as e:
                        logger.warning(f"Skipping row due to error: {e}")
                        continue
                
                logger.info(f"Successfully loaded {len(self.people)} people")
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def get_todays_birthdays(self) -> List[Dict]:
        """Return people with birthdays today"""
        today = datetime.now()
        return [
            person for person in self.people
            if person.get('Birthday Month') == str(today.month) 
            and person.get('Birthday Day') == str(today.day)
        ]
    
    def get_suggestions(self, count: int = 2) -> List[Dict]:
        """Get weighted random suggestions"""
        candidates = [
            person for person in self.people
            if self.user_connections.get(person['Name']) != "Stop"
        ]
        
        # Apply weights
        weighted = []
        for person in candidates:
            level = self.user_connections.get(person['Name'], "Normal")
            weighted.extend([person] * self.weights.get(level, 1))
            
        return random.sample(weighted, min(count, len(weighted)))
    
    def update_connection(self, name: str, level: str) -> bool:
        """Update connection level for a person"""
        if level not in self.weights:
            return False
            
        self.user_connections[name] = level
        logger.info(f"Updated {name} to {level}")
        return True
