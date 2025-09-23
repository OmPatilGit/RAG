from neo4j import GraphDatabase

def checkConnectivity(URI : str, UNAME : str, PASSED : str):
    driver = GraphDatabase.driver(URI, auth=(UNAME, PASSED))

    if driver.verify_connectivity():
        print("Connection Successful !")
        return True, driver
    
    print("Connection Unsuccessful !")
    return False
