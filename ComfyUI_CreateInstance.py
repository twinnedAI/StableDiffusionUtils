#!pip install -r requirements.txt
import json
import time
from vastai import VastAI
import argparse

parser = argparse.ArgumentParser("Vastai create ComfyUI instance")
parser.add_argument('civit_token', type=str, help="Civit.ai token for provisioning scripts")
parser.add_argument('hf_token', type=str, help="Huggingface token for provisioning scripts")
parser.add_argument('-dry', action='store_true', help="Set this flag to not create an instance")
parser.add_argument('-copy', action='store_true', help="Set this flag to copy cloud data to the created instance")
parser.add_argument('-onlyCopy', action='store_true', help="Set this flag to only copy cloud data to an existing instance")
args = parser.parse_args()

CIVIT_AI_TOKEN=parser.civit_token
HF_TOKEN=parser.hf_token

# some search params
dph = 0.38
gpu_ram = 24
gpu_names= "[\"RTX_4090\"]"
disk_space=100

# connections
cloud_data_connection_id="13808"
provisioning_script="https://gist.github.com/twinnedAI/0e0cb62e50a19b467ad7df6e41dd14ca/raw"
#provisioning_script="https://gist.github.com/twinnedAI/5f3db778097c3f622aa392e891bbf695/raw"
#"https://gist.github.com/twinnedAI/98a1acfe548c649a643def2317ac7e05/raw"
client = VastAI(api_key='b921f4caa3e3eb8fbadf23056b03a95a751e4ad8bda2ff7ac6a0c8fc062775bf', raw=True)

def loadInstances():
    instances_raw = client.show_instances()
    return json.loads(instances_raw)

def loadInstanceById(instance_id):
    instance_raw = client.show_instance(id=instance_id)
    return json.loads(instance_raw)

def get_service_portal_url(instance):
    pub_ip = instance.get('public_ipaddr')
    service_portal_port = instance.get('ports').get('1111/tcp')[0].get('HostPort')
    return f"http://{pub_ip}:{service_portal_port}"

def copyCloudData(instance):
    print(F"Start copy checkpoints")
    client.cloud_copy(src='/workspace/ComfyUI/models/checkpoints', dst='/opt/ComfyUI/models/checkpoints',instance=instance.get('id'), connection='13808', transfer="Cloud To Instance")
    instance_ready = False
    while not instance_ready:
        instance = loadInstanceById(instance_id=instance.get('id'))
        if (instance.get("status_msg") == "Cloud Copy Operation Complete\n"):
            instance_ready = True
        else:
            print(f"Instance not ready... status_msg: {instance.get("status_msg")} status: {instance.get('actual_status')} intended_status: {instance.get('intended_status')}")
            time.sleep(10)
    print(F"Start copy loras")
    client.cloud_copy(src='/workspace/ComfyUI/models/loras', dst='/opt/ComfyUI/models/loras',instance=instance.get('id'), connection=cloud_data_connection_id, transfer="Cloud To Instance")

def create_contract():
    query = F'dph>={dph} reliability>0.95 num_gpus=1 verified=true gpu_ram>={gpu_ram} disk_space>={disk_space} gpu_name in {gpu_names} geolocation in [DE, FR, PL, SE, ES, PT, CH, NO, LU, CZ, US]'
    #print(query)
    offers_raw = client.search_offers(query=query,order='dph+', limit=8)
    offers = json.loads(offers_raw)
    if not offers:
        print(F'No offers found for query: {query}')
        return None
    top_offer = offers[0]
    #print(f"Found top offer:")
    #print(json.dumps(top_offer, indent=2))
    top_offer_id = top_offer.get('id')
    print(f"Going to create instance for offer {top_offer_id}")
    if not args.dry:
        #contract_raw = client.create_instance(ID=int(top_offer_id), image='ghcr.io/ai-dock/comfyui:latest', ssh=True, disk=65, env=f"-p 1111:1111 -p 8188:8188 -e AUTO_UPDATE=true -e WEB_USER=user -e WEB_PASSWORD=user123 -e PROVISIONING_SCRIPT={provisioning_script}", onstart_cmd='env | grep _ >> /etc/environment; /opt/ai-dock/bin/init.sh;')
        contract_raw = client.create_instance(ID=int(top_offer_id), image='ghcr.io/ai-dock/comfyui:latest', jupyter=True, disk=65, env=f"-p 22:22 -p 1111:1111 -p 8188:8188 -e CIVIT_AI_TOKEN={CIVIT_AI_TOKEN} -e HF_TOKEN={HF_TOKEN} -e WEB_USER=user -e WEB_PASSWORD=user123 -e AUTO_UPDATE=true -e FOOOCUS_FLAGS=\"--preset realistic\" -e DATA_DIRECTORY=/workspace -e WORKSPACE=/workspace -e WORKSPACE_MOUNTED=force -e FOOOCUS_BRANCH=main -e OPEN_BUTTON_TOKEN=1 -e OPEN_BUTTON_PORT=1111 -e PROVISIONING_SCRIPT={provisioning_script}", onstart_cmd='env | grep _ >> /etc/environment; /opt/ai-dock/bin/init.sh;')
        return json.loads(contract_raw)
    else:
        return None

def prepare_instance(instance):
    instance_ready = False
    while not instance_ready:
        instance = loadInstanceById(instance_id=instance_id)
        if (instance.get('ports') and instance.get('actual_status') == instance.get('intended_status')):
            instance_ready = True
        else:
            print(f"(Waiting...) Instance not ready... ports: {instance.get('ports')} status: {instance.get('actual_status')} intended_status: {instance.get('intended_status')}")
            time.sleep(10)
    instance = loadInstanceById(instance_id=instance_id)
    print(F"Instance is ready! Visit: {get_service_portal_url(instance)}")
    if args.copy:
        time.sleep(60)
        copyCloudData(instance)
    print(F"Cloud data copied! Visit: {get_service_portal_url(instance)}")

if args.onlyCopy:
    existing_instance = loadInstances()[0]
    copyCloudData(existing_instance)
else:
    existing_instances = loadInstances()
    instance = None

    if existing_instances and len(existing_instances) > 0:
        print(F"Existing instance ({existing_instances[0].get('id')}) detected. Skip...")
    else:
        print(F"No existing instance found. Create new contract")
        contract = create_contract()
        if contract and contract.get('success') == True:
            instance_id = contract.get('new_contract')
            print(f"Created contract and instance with id {instance_id}")
            instance = loadInstanceById(instance_id=instance_id)
            prepare_instance(instance=instance)
        else:
            print("Failed to create instance")
            print(json.dumps(instance, indent=2))
