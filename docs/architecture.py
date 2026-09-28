"""Architecture diagram for saranreddy/sagemaker-mlops-pipeline-starter.

Verified against main @ ceea5d0. Every node maps to infra/*.tf, src/pipeline/pipeline.py
or scripts/*.py.

Render:  pip install diagrams   (also needs Graphviz: apt install graphviz / brew install graphviz)
         python docs/architecture.py   ->  docs/architecture.png (written next to this script)
"""
import os

from diagrams import Cluster, Diagram, Edge, getdiagram
from diagrams.aws.compute import EC2ContainerRegistryImage
from diagrams.aws.general import User
from diagrams.aws.management import CloudwatchLogs
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import SimpleStorageServiceS3Bucket
from diagrams.onprem.iac import Terraform
from diagrams.programming.flowchart import Decision
from diagrams.programming.language import Python

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "architecture")  # -> architecture.png next to this script

FONT = "DejaVu Sans"
GRAPH = {
    "fontname": FONT, "fontsize": "34", "labelloc": "t", "pad": "0.4",
    "nodesep": "0.4", "ranksep": "1.0", "splines": "spline", "newrank": "true",
    "compound": "true",
}
NODE = {"fontname": FONT, "fontsize": "21", "imagepos": "tc"}
EDGE = {"fontname": FONT, "fontsize": "19", "color": "#555555",
        # enter/leave icons at mid-height so arrowheads never land on label text
        "tailport": "e", "headport": "w"}

# diagrams.Edge hard-codes a 13pt label font on every edge; raise it so edge labels stay
# readable when the PNG is scaled down to README width.
Edge._default_edge_attrs = {"fontcolor": "#2D3436", "fontname": FONT, "fontsize": "19"}


def box(bg, pen, style="rounded"):
    return {"bgcolor": bg, "pencolor": pen, "fontname": FONT, "fontsize": "21",
            "style": style, "labeljust": "l", "margin": "24"}


TF_BOX = box("#fff4e0", "#e66100")                  # deployed by Terraform
SUB_BOX = box("#fffaf2", "#e66100")                 # sub-group inside a Terraform box
RUN_BOX = box("#e8f1fb", "#1a5fb4")                 # created by scripts / CLI
EXEC_BOX = box("#f3eefa", "#613583")                # per execution / runtime
MANAGED_BOX = box("#f6f5f4", "#9a9996", "dashed")   # not created by this repo
ACCOUNT_BOX = box("#ffffff", "#232f3e")

FLOW = dict(color="#1a5fb4", fontcolor="#1a5fb4", penwidth="2.2")
IO = dict(color="#26a269", fontcolor="#1e7d4f", penwidth="1.8")
IAM = dict(color="#c01c28", fontcolor="#c01c28", style="dashed", penwidth="1.6", constraint="false")
AUX = dict(color="#8a8a8a", fontcolor="#5e5c64", style="dotted", penwidth="1.8")
SETUP = dict(color="#e66100", fontcolor="#c64600", style="dashed", penwidth="1.8")
MANUAL = dict(color="#26a269", fontcolor="#1e7d4f", style="dashed", penwidth="2.2")
FAIL = dict(color="#c01c28", fontcolor="#c01c28", penwidth="2.2")
OPT = dict(color="#b5835a", fontcolor="#8f5f3a", style="dashed", penwidth="1.8")
HIDDEN = dict(style="invis")
DOWN = dict(tailport="s", headport="n")
UP = dict(tailport="n", headport="s")


def same_rank(*nodes):
    getdiagram().dot.body.append("{rank=same; " + " ".join(f'"{n._id}";' for n in nodes) + "}")


# Top-to-bottom layout keeps the 5-step pipeline on one row, so the PNG stays narrow
# enough to read at README width.
GRAPH["ranksep"] = "0.9"
GRAPH["pad"] = "0.8"
GRAPH["nodesep"] = "1.3"
GRAPH["forcelabels"] = "true"
EDGE_TB = {k: v for k, v in EDGE.items() if k not in ("tailport", "headport")}
ROW = dict(tailport="e", headport="w")     # flat edge inside a row, left -> right

with Diagram(
    "sagemaker-mlops-pipeline-starter",
    filename=OUT, outformat="png", show=False, direction="TB",
    graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE_TB,
):
    eng = User("ML engineer")
    sdk = Python("upsert_pipeline.py\nstart_pipeline.py\n(SageMaker SDK)")
    tf = Terraform("terraform apply\n(infra/)")

    with Cluster("AWS account  (default us-east-1)", graph_attr=ACCOUNT_BOX):

        with Cluster("Created by the scripts", graph_attr=RUN_BOX):
            sdkbucket = SimpleStorageServiceS3Bucket("SDK bucket\n(step code)\nsagemaker-\n<region>-<acct>")
            pipeline = Sagemaker("Pipeline\ncalifornia-\nhousing-\npipeline")

        with Cluster("Deployed by Terraform", graph_attr=TF_BOX):
            # (declaration order tuned so graphviz lays out role | bucket | group)
            mpg = SagemakerModel("Model package\ngroup\ncalifornia-\nhousing-models")
            bucket = SimpleStorageServiceS3Bucket("Artifact bucket\nversioned,\nSSE-S3")
            role = IAMRole("SageMaker\nexecution role")

        with Cluster("Per pipeline execution", graph_attr=EXEC_BOX) as run:
            prep = Sagemaker("PreprocessData\n(downloads the\ndataset via\nsklearn)")
            train = SagemakerTrainingJob("TrainModel\nXGBoost 1.7-1")
            evaluate = Sagemaker("EvaluateModel\nprocessing job")
            gate = Decision("CheckMse\nThreshold\nMSE <= 0.5\nelse: no model")
            version = SagemakerModel("RegisterModel\nPending\nManualApproval")

        with Cluster("Created by deploy_model.py", graph_attr=RUN_BOX):
            endpoint = Sagemaker("Real-time\nendpoint\nml.t2.medium")
            epc = Sagemaker("Endpoint config")
            model = SagemakerModel("SageMaker Model")

        with Cluster("AWS-managed", graph_attr=MANAGED_BOX):
            ecr = EC2ContainerRegistryImage("sklearn /\nXGBoost\nframework\nimages")
            logs = CloudwatchLogs("CloudWatch\nLogs /aws/\nsagemaker/*")

    deploy = Python("deploy_model.py\n(boto3)")
    approver = User("ML engineer\n(approve + deploy)")

    # row 1: actors and tools
    eng >> Edge(label="1. run", **FLOW) >> sdk
    eng >> Edge(label="deploy", **SETUP) >> tf
    tf >> Edge(**SETUP) >> role
    tf >> Edge(**SETUP) >> bucket
    tf >> Edge(**SETUP) >> mpg
    same_rank(pipeline, sdkbucket, role, bucket, mpg)

    # row 2 -> row 3: pipeline
    sdk >> Edge(label="upsert + start", **FLOW) >> pipeline
    sdk >> Edge(label="upload step code", **IO) >> sdkbucket
    role >> Edge(label="assumed", **IAM) >> pipeline
    pipeline >> Edge(label="runs", **FLOW) >> prep
    sdkbucket >> Edge(**IO) >> prep
    prep >> Edge(xlabel="train / val /\ntest CSV", **ROW, **FLOW) >> train
    train >> Edge(xlabel="model\n.tar.gz", **ROW, **FLOW) >> evaluate
    evaluate >> Edge(xlabel="evaluation\n.json", **ROW, **FLOW) >> gate
    gate >> Edge(xlabel="pass", color="#26a269", fontcolor="#1e7d4f", penwidth="2.2", **ROW) >> version
    same_rank(prep, train, evaluate, gate, version)
    bucket << Edge(label="step outputs", lhead=run.name, **IO) << evaluate
    mpg << Edge(label="version in", style="dashed", tailport="s", headport="n", **FLOW) << version

    # row 4: approval + deployment (reads right-to-left, towards the engineer's script)
    approver >> Edge(label="2. approve", constraint="false", **MANUAL) >> version
    # Leftward flat edges: drawn as forward edges with w/e ports, while invisible
    # left-to-right edges pin the order (endpoint, config, model | deploy, approver).
    LEFT = dict(tailport="w", headport="e", constraint="false")
    endpoint >> Edge(**HIDDEN) >> epc
    epc >> Edge(**HIDDEN) >> model
    model >> Edge(**HIDDEN) >> deploy
    deploy >> Edge(**HIDDEN) >> approver
    approver >> Edge(xlabel="3. run", **LEFT, **MANUAL) >> deploy
    mpg >> Edge(label="latest Approved\n(else Pending)", tailport="e", constraint="false", **FLOW) >> deploy
    deploy >> Edge(xlabel="create", **LEFT, **FLOW) >> model
    model >> Edge(**LEFT, **FLOW) >> epc
    epc >> Edge(**LEFT, **FLOW) >> endpoint
    version >> Edge(**HIDDEN) >> approver

    # supporting
    prep >> Edge(**HIDDEN) >> ecr
    train >> Edge(**HIDDEN) >> logs
    gate >> Edge(**HIDDEN) >> model
    ecr >> Edge(constraint="false", **AUX) >> train
    train >> Edge(constraint="false", **AUX) >> logs
    same_rank(ecr, logs, model, epc, endpoint, approver, deploy)
