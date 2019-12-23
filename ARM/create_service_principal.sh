#az cli script for installing ARM template - in progress
read -p "Enter name for service principal user (required for de-allocating VM after the tasks are complete): " serviceprincipal_name
read -p "Enter your subscriptionId: " subID

#check if service principal exists
# if [ -z "$var" ]
# then
# echo "\$var is NULL"
# else
# echo "\$var is NOT NULL"
# fi

serviceprincipal_details=$(az ad sp create-for-rbac -n "$serviceprincipal_name" 
--role "Virtual Machine Contributor" --scopes /subscriptions/$SubID/ --query ['tenant','appId','password'] --o tsv )

serviceprincipal_details_list=($serviceprincipal_details)
tenantId={serviceprincipal_details_list[0]}
appID=${serviceprincipal_details_list[1]}
password=${serviceprincipal_details_list[2]}

echo "Save the following details you will need this information in the create VM step"

echo "appId: $appID"
echo "tenantId: $tenantId"
echo "password: $password"
