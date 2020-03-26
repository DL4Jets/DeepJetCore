
#not used for now


def renew_token_process():
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
    return 
    
    import subprocess
    import os
    
    klist=""
    try:
        os.environ['LC_ALL']="en_US.UTF-8"
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
    month,day,year=kdate.split('/')
    hour,minu,sec=ktime.split(':')
    try:
        tokentime=datetime.datetime(2000+int(year),int(month),int(day),int(hour))
    except:
        print('Failed to set token time with mm/dd/yy, attempting dd/mm/yy permutation')
        tokentime=datetime.datetime(2000+int(year),int(day),int(month),int(hour))

    diff=tokentime-thistime
    diff=diff.total_seconds()
    
    if diff < cutofftime_hours*3600:
        print('token will expire soon. Starting kinit')
        subprocess.check_call(['kinit','-l 96h'])
        subprocess.check_call(['aklog'])
    return True
    
