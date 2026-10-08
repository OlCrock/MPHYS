import yt

path = "/cephfs2/brs/bw_mw/100Mpc_256/dm-only-L0/DD0000/DD0000"

ds = yt.load(path)

print(ds)
print("Redshift:", ds.current_redshift)
print("Domain width:", ds.domain_width)
print("Domain left:", ds.domain_left_edge)
print("Domain right:", ds.domain_right_edge)

print("\nParticle fields:")
print(ds.field_list)