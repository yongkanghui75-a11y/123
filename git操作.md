



🧠一、文件 / 目录操作（最常用）
pwd              # 查看当前路径
ls               # 查看文件
ls -l            # 详细列表
ls -a            # 显示隐藏文件

cd 目录名         # 进入目录
cd ..            # 返回上一级
cd ~             # 回到家目录

mkdir test       # 创建文件夹
touch a.txt      # 创建文件
rm a.txt         # 删除文件
rm -r folder     # 删除文件夹
cp a b           # 复制
mv a b           # 移动/重命名
🧠 二、文件内容查看
cat file.txt     # 查看全文
less file.txt    # 翻页查看
head file.txt    # 前10行
tail file.txt    # 后10行
🧠 三、系统 & 进程
top              # 任务管理器
htop             # 更好看的（需安装）
ps aux           # 查看进程
kill PID         # 杀进程
df -h            # 磁盘空间
free -h          # 内存
🧠 四、软件安装（Ubuntu核心）
sudo apt update          # 更新软件列表
sudo apt upgrade         # 升级软件
sudo apt install xxx     # 安装软件
sudo apt remove xxx      # 删除软件
🧠 五、权限相关（你以后一定会用）
sudo xxx         # 用管理员执行
chmod +x file    # 赋予执行权限
chown user file   # 修改所有者
🧠 六、压缩 / 解压
tar -xvf file.tar
tar -czvf a.tar.gz folder
unzip file.zip
🧠 七、Git（你现在最重要）
git init                 # 初始化仓库
git status              # 查看状态
git add .               # 添加所有修改
git commit -m "msg"     # 提交版本
git push                # 上传GitHub
git pull                # 拉取更新

git log                 # 查看历史
git diff                # 查看修改
🚀 Git远程相关
git remote -v                     # 查看远程
git remote add origin URL        # 添加远程
git remote set-url origin URL    # 修改远程
🧠 八、SSH（你刚刚学的）
ssh-keygen -t ed25519 -C "email"   # 生成密钥
eval "$(ssh-agent -s)"             # 启动ssh
ssh-add ~/.ssh/id_ed25519          # 添加密钥

cat ~/.ssh/id_ed25519.pub          # 查看公钥
🧠 九、网络 / 下载
ping google.com
curl URL
wget URL
🧠 十、环境 & Python（科研常用）
python3 xxx.py
pip install package

conda activate env_name
conda list
🧠 十一、超级常用组合（重点）
✔ 看状态 + 提交 + 上传
git status
git add .
git commit -m "update"
git push
✔ 新项目完整流程
mkdir project
cd project
git init
git add .
git commit -m "init"
🧠 十二、git上传

git add 文件名                  / //添加
git commit -m 'update commit'      //提交
git commit -a -m 'update commit'                           //添加并提交
git log                                               //查看历史版本号
git reset 版本号                                           //恢复到某个版本号
git rm 文件名                                     //删除
git status                                //查看状态


更新（第一次需跟github仓库连接）
git add .
git commit -m "本次修改说明"
git push




