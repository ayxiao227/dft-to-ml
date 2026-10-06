# import paramiko

# hostname = "greatlakes.arc-ts.umich.edu"
# username = "xaustin"
# path = "/home/xaustin/dft_ml/"

# #scp -r ./molecules/gjf_files xaustin@greatlakes.arc-ts.umich.edu:~/dft_ml

# ssh = paramiko.SSHClient()
# ssh.load_system_host_keys()
# ssh.connect(hostname, username=username)

# stdin, stdout, stderr = ssh.exec_command("pwd")

# print(stdout.read().decode())


# sftp.put(
#     "molecules/gjf_files/methane.gjf",
#     path + "methane.gjf"
# )

# sftp.close()
# ssh.close()


