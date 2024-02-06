from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load 

# Créer une instance FastAPI
app = FastAPI()

# Définir la structure de la requête avec Pydantic
class InputData(BaseModel):
    NumberofFloors : int  
    PropertyGFATotal: float
    SourceEUI : float
    SteamUse : float
    Electricity : float 
    NaturalGas: float
    NumberOfUse: int
    BuildingAge: int
    BuildingType:object
    PrimaryPropertyType: object 
    Neighborhood:object

# Chargeement du modèle choisi
loaded_model = load('modele.joblib')  

# Endpoint pour prédire avec le modèle
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convertir les données d'entrée en tableau numpy ou pandas selon votre modèle
        input_features = [[data.NumberofFloors, data.PropertyGFATotal, data.SourceEUI, data.SourceEUI, 
                           data.SteamUse, data.Electricity, data.NaturalGas, data.NumberOfUse, data.BuildingAge,
                           data.BuildingType, data.PrimaryPropertyType, data.Neighborhood]]  

        # Faire la prédiction avec le modèle chargé
        prediction = loaded_model.predict(input_features)

        # Retourner le résultat de la prédiction
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Endpoint pour vérifier que l'API est en ligne
@app.get("/")
def read_root():
    return {"message": "L'API est en ligne!"}
