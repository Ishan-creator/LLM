import pymongo
import datetime
import pprint



client = pymongo.MongoClient('mongodb+srv://ishanpokhrel11:Ch7w9jsbFMAHokn1@cluster0.fj91ksq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

collection = client['gettingStarted']['people']



# Create new documents
peopleDocuments = [
    {
      "name": { "first": "Alan", "last": "Turing" },
      "birth": datetime.datetime(1912, 6, 23),
      "death": datetime.datetime(1954, 6, 7),
      "contribs": [ "Turing machine", "Turing test", "Turingery" ],
      "views": 1250000
    }, 
    {
      "name": { "first": "Grace", "last": "Hopper" },
      "birth": datetime.datetime(1906, 12, 9),
      "death": datetime.datetime(1992, 1, 1),
      "contribs": [ "Mark I", "UNIVAC", "COBOL" ],
      "views": 3860000
    }
]

# Insert documents
collection.insert_many(peopleDocuments)

# Insert one document
collection.insert_one({
      "name": { "first": "Ishan", "last": "Pokhrel" },
      "birth": datetime.datetime(2002, 2, 17),
      "death": datetime.datetime(3002, 6, 7),
      "contribs": [ "Nothing" ],
      "views": 0
})

# Read Operation
result = collection.find_one({ "name.last": "Pokhrel" })

if result:
    pprint.pprint(result)
else:
    print("Document not found")

# Update Operation
update_result = collection.update_one(
    {"name.first": "Ishan"},
    {
        "$set": {"views": 1000}
    }
)


# Read Operation after update to verify
result = collection.find_one({ "name.first": "Ishan" })

if result:
    pprint.pprint(result)
else:
    print("Document not found")
    
    
#Delete operation
delete_result = collection.delete_one({"name.first": "Ishan"})

#Delete Operaton Many 
delete_result = collection.delete_many({})