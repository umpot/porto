sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 0C49F3730359A14518585931BC711F9BA15703C6
echo "deb [ arch=amd64 ] http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.4.list
sudo apt-get update
sudo apt-get install -y mongodb-org

#Edit host - set local IP i.e. 10.138.0.2 NOT GLOBAL
sudo vim /etc/mongod.conf

#start service
sudo service mongod start

#Check errors in log
vim /var/log/mongodb/mongod.log

#security
#mongo
use admin
db.createUser(
  {
    user: "ubik",
    pwd: "nfrf[eqyz",
    roles: [ { role: "root", db: "admin" } ]
  }
)

#One more fucking bullshit:
use admin
db.auth("ubik", "nfrf[eqyz")
db.grantRolesToUser("ubik", [ { role: "readWrite", db: "admin" } ])

#Enable security in mongod.conf
security:
	authorization: enabled