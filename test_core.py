from main import ConnectionManager
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_csv_file():
    """Verify CSV file exists and is readable"""
    csv_file = "SFMBA Directory.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return False
    
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle BOM
            line = f.readline()
            print(f"CSV first line: {line[:100]}...")
        return True
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

# Test the core functionality
def test_connection_manager():
    print("\n=== Testing ConnectionManager ===")
    
    if not check_csv_file():
        return
    
    try:
        manager = ConnectionManager()
        print(f"\nLoaded {len(manager.people)} people")
        
        if not manager.people:
            print("Warning: No people loaded from CSV")
            return
            
        # Show first person as sample
        sample = manager.people[0]
        print(f"\nSample person:")
        print(f"Name: {sample['Name']}")
        print(f"Email: {sample['Personal Email']}")
        print(f"Birthday: {sample['Birthday Month']}/{sample['Birthday Day']}")
        
        # Test birthday check
        today = datetime.now()
        print(f"\nToday is {today.month}/{today.day}")
        birthdays = manager.get_todays_birthdays()
        print(f"Found {len(birthdays)} birthdays today")
        
        # Test suggestions
        suggestions = manager.get_suggestions(count=2)
        print(f"\nSuggestions (count: {len(suggestions)}):")
        for person in suggestions:
            print(f"- {person['Name']}")
        
        # Test updating connection
        if suggestions:
            name = suggestions[0]['Name']
            print(f"\nUpdating {name} to 'Close'")
            manager.update_connection(name, "Close")
            print(f"Updated {name}: {manager.user_connections.get(name)}")
        
        print("\nTests completed!")
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    test_connection_manager()
