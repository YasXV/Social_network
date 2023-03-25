import numpy as np
import sqlite3

#this code counts the number of interactions between 2 ids within "tij_Thiers13.dat" file 
con=sqlite3.connect("class.db")
cur=con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS class(time,id1,id2)")
data=np.loadtxt("tij_Thiers13.dat")
res=cur.execute("SELECT * FROM class")
if res.fetchall()==[]:
    cur.executemany("INSERT INTO class VALUES(?,?,?)",data)
    con.commit()
res2=cur.execute("SELECT id1,id2,COUNT(time) FROM class GROUP BY id2,id1 order by id1")
data2=np.array(res2.fetchall())
np.savetxt("tij_tri.dat",data2,fmt='%4i',header='id1  id2 numb_interact')
con.close()
