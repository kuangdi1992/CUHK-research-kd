在相关的目录下(现在我的是代码)，使用：  
1.git add 文件名字  
2.git commit -m "kuangdi"  
3.git remote add origin git@github.com:kuangdi1992/code.git  
4.git push -u origin master  
一个设置ssh的网址：https://blog.csdn.net/jenyzhang/article/details/44703937  


如果出现了这样的问题：  
[kd@kd-pc 代码]$ git push -u origin master  
ERROR: Repository not found.  
fatal: 无法读取远程仓库。  
解决办法：  git remote set-url origin git@github.com:kuangdi1992/codes.git  

今天想将代码链接到远程仓库中，但是出现了这样的报错：   
fatal: 不是一个 git 仓库（或者任何父目录）：.git  
查找后解决方案是：git init  

之后可以按照下面的方法来做： https://www.openfoundry.org/tw/foss-programs/9318-git-github-firsttry  

git fatal: 远程 origin 已经存在。  
此时只需要将远程配置删除，重新添加即可；  
git remote rm origin  
git remote add origin https://github.com/***/WebCrawlers.git  
再次提交文件即可正常使用  



问题：文件过大，留在了commit中上传不上去？<br>
解决方法：找到之前的commit的号，git log，然后利用git reset --hard　<commit号>，即可删除之前的历史记录。<br>
具体如下：
<pre><code>
➜  codes git:(master) git push -u origin master
枚举对象: 95, 完成.
对象计数中: 100% (95/95), 完成.
使用 4 个线程进行压缩
压缩对象中: 100% (90/90), 完成.
写入对象中: 100% (94/94), 854.86 MiB | 1.58 MiB/s, 完成.
总共 94 （差异 32），复用 0 （差异 0）
remote: Resolving deltas: 100% (32/32), completed with 1 local object.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 35eb19e2aad9db5a3c055f1de4dd0e88
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File cifar10/data/cifar-10-python.tar.gz is 162.60 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl is 488.91 MB; this exceeds GitHub's file size limit of 100.00 MB
To github.com:kuangdi1992/codes.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: 推送一些引用到 'git@github.com:kuangdi1992/codes.git' 失败
➜  codes git:(master) git log
➜  codes git:(master)
➜  codes git:(master) git reset --hard 1d197070b441fc2dc138e7588f61a69038a4a6da
HEAD 现在位于 1d19707 didi
➜  codes git:(master) git add .
➜  codes git:(master) ✗ git commit -m "didi"
[master 6a227cd] didi
 1 file changed, 245 insertions(+)
 create mode 100644 "\347\254\224\350\256\260.md"
➜  codes git:(master) git push -u origin master
To github.com:kuangdi1992/codes.git
 ! [rejected]        master -> master (non-fast-forward)
error: 推送一些引用到 'git@github.com:kuangdi1992/codes.git' 失败
提示：更新被拒绝，因为您当前分支的最新提交落后于其对应的远程分支。
提示：再次推送前，先与远程变更合并（如 'git pull ...'）。详见
提示：'git push --help' 中的 'Note about fast-forwards' 小节。
➜  codes git:(master) git pull origin master
来自 github.com:kuangdi1992/codes
 * branch            master     -> FETCH_HEAD
Merge made by the 'recursive' strategy.
 "2018\345\271\26412\346\234\21020\346\227\245\347\254\224\350\256\260.md" | 75 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 1 file changed, 73 insertions(+), 2 deletions(-)
➜  codes git:(master) git push -u origin master
枚举对象: 7, 完成.
对象计数中: 100% (7/7), 完成.
使用 4 个线程进行压缩
压缩对象中: 100% (5/5), 完成.
写入对象中: 100% (5/5), 4.33 KiB | 4.33 MiB/s, 完成.
总共 5 （差异 2），复用 0 （差异 0）
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
To github.com:kuangdi1992/codes.git
   12c0713..31791f9  master -> master
分支 'master' 设置为跟踪来自 'origin' 的远程分支 'master'。
</code></pre>
