## CLOUDE CONCEPTS

# Cloud Concepts Question 1

What is the core economic model of cloud computing, and how does it differ from owning your own servers?  Is cheaper a cloud computing, because you don’t have to own the equipment, is a rent for a service and could be customize it what is needed.  Owning the own servers is more expensive, not just the equipment but also to maintain them and have layers of security.  Also, owning the own servers is limited the capability while with cloud computing can be accessed many servers.


# Cloud Concepts Question 2

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling is when need to expand the equipment capabilities, like memory, disk, etc.  Horizontal scaling is when is needed to have multiple machines.

Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.  Horizontal scaling
A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.  Vertical scaling
A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.  Horizontal scaling


# Cloud Concepts Question 3

Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.

Gmail  SaaS, we access the service no need to setup the functionality.
Azure Virtual Machines  IaaS, even though is virtual we need to configure the environment and setup the machine.
Azure App Service  PaaS, we need to deploy an application or code so the platform can run it.
AWS S3 (Simple Storage Service)  IaaS, it’s an infrastructure that provides storage doesn’t provide a software application
GitHub Codespaces  PaaS, since it is from GitHub it is in the cloud, is a platform that allow to build and run code
Snowflake  SaaS, I think is SaaS because provides a service as cloud application.
Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

IaaS, provides an infrastructure.  Ex: AWS EC2, provides a virtual machine.

PaaS, is a platform that lets to build on top of it. Ex.: Heroku, provides the platform to develop code

SaaS, provides a service through a website.  It could be Google Drive, it requires a login to access the service..

 
# Cloud Concepts Question 4

What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

A managed data platform is a cloud infrastructure that is setup for data and analytics workloads.  With the managed data platform we gain that is much faster to get started with large-scale data processing or machine learning, at the cost of some flexibility and potentially higher cost.  Cloud providers provides the full toolkit: compute, storage, networking, machine learning services, databases,etc.

 

Cloud Concepts Question 5

The lesson names two situations where the cloud is probably not the right choice. What are they? There are the learning curve and the support also we can add that the cost could spiral fast.

Learning curve can be very steep, it requires to figure out the right resources and jargon initially.  The support could take time or could be outdated, that makes it difficult to have the right support.  If the cluster runs overnight or forget to clean up storage can generate a high cost.

 
## AZURE BASICS

These questions are based on the Getting Started with Azure lesson.


# Azure Basics Question 1

What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?

The difference is that the Azure subscription belongs to an organization and a resource group is what each student gets.


# Azure Basics Question 2

Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

The ephemeral by default means that every time the shell is closed all the files and directories are deleted.    To make it persistent the instructors created a file share connected to a Cloud Shell which is a storage folder in Azure similar to a network drive.


# Azure Basics Question 3

What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

The SSH private key stays on the machine while the public key is uploaded to the systems we want to access.  To the remote systems gets uploaded the public key, is safe because is an encrypted protocol that communicates securely between computers or over a network.

 
# Azure Basics Question 4

Run the following command in Cloud Shell without the --output table flag:

az account show

Paste the output into your answer. Then describe in one sentence what changes when you add --output table.

"homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#marvin.diaz1@gmail.com",
    "type": "user"
  }
}

TABLE
marvin [ ~ ]$ az account show --output table
EnvironmentName    HomeTenantId                          IsDefault    Name                       State    TenantId
-----------------  ------------------------------------  -----------  -------------------------  -------  ------------------------------------
AzureCloud         0f040ddd-301f-4665-8677-7b21f129d605  True         CTD Nonprofit Sponsorship  Enabled  0f040ddd-301f-4665-8677-7b21f129d605

# When added THE --OUTPUT TABLE CONFIGURE THE INFORMATION IN TABULAR (TABLE) SHAPE
