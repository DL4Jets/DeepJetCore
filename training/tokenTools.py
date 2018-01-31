
renewtokens=True

def renew_token_process():
    
    if not renewtokens:
        return 
     
    import subprocess
    import time 
    while True:
        logstr=""
        try:
            logstr=subprocess.check_call(['kinit', '-R'])
            logstr+=subprocess.check_call(['aklog'])
        except:
            print(logstr)
        time.sleep(3600)

def checkTokens(cutofftime_hours=48):
    if not renewtokens:
        return True
    import subprocess
    
    klist=""
    try:
        klist=str(subprocess.check_output(['klist'],stderr=subprocess.STDOUT))
    except subprocess.CalledProcessError as inst:
        print('klist failed - no token?')#just ignore
        klist=""
        del inst
        
    
    if not 'renew' in klist:
        print('did not find renew option in kerberos token. Starting kinit')
        subprocess.check_call(['kinit','-l 96h'])
        subprocess.check_call(['aklog'])
        return True
        
    klist=str(klist).split()
    
    firstrenewapp=klist.index('renew')
    
    
    kdate=klist[firstrenewapp+2]
    ktime=klist[firstrenewapp+3]
    
    
    import datetime
    thistime=datetime.datetime.now()
    tokentime=datetime.datetime(2000+int(kdate.split('/')[2]) ,
                                int(kdate.split('/')[0]),
                                int(kdate.split('/')[1]),
                                int(ktime.split(':')[0]))
    diff=tokentime-thistime
    diff=diff.total_seconds()
    
    if diff < cutofftime_hours*3600:
        print('token will expire soon. Starting kinit')
        subprocess.check_call(['kinit','-l 96h'])
        subprocess.check_call(['aklog'])
    return True
    
