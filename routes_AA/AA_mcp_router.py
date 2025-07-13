from fastapi import APIRouter, UploadFile
from server_tools.topology_analyzer_OLD import TopologyAnalyzer

router = APIRouter()
analyzer = TopologyAnalyzer()

@router.post("/analyze")
async def analyze_topology(image: UploadFile, query: str):
    topology = await image.read()
    return analyzer.analyze(topology, query)

@router.post("/transform")
async def transform_topology(image: UploadFile, target_vendor: str = "aruba"):
    topology = await image.read()
    return analyzer.transform(topology, target_vendor)
