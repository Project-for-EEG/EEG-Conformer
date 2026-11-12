import re,html,sys
t=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
t=re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S)
t=re.sub(r'<[^>]+>','\n',t); t=html.unescape(t)
lines=[l.strip() for l in t.split('\n') if l.strip()]
s='\n'.join(lines)
start=int(sys.argv[2]) if len(sys.argv)>2 else 0
n=int(sys.argv[3]) if len(sys.argv)>3 else 10000
sys.stdout.buffer.write(s[start:start+n].encode('utf-8'))
